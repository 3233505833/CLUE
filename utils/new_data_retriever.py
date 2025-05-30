from utils.doc_use_tuple import get_doc_use_tuple
import json
from collections import OrderedDict
from typing import Union, List, Tuple

import faiss
from utils.new_file_utils import DataItem
from collections import namedtuple

from tqdm import tqdm
from transformers import AutoConfig, AutoModel
import torch
assert hasattr(faiss, "IndexFlatIP")
import numpy as np
from utils.new_data_utils import Tokenizer, encode_md5hash
from utils.new_data_loader import AbsDataloader
from utils.new_file_utils import check_file_and_mkdir_for_save, load_jsonl
class AbsDataRetriever(object):
    def __init__(self, device: str = "cpu", num_cells_in_search: int = 10, num_cells: int = 100, logger=None,
                 saved_nearest_neighbor_file: str = None):
        self.logger = ""
        self.device = device
        self.num_cells_in_search = num_cells_in_search
        self.num_cells = num_cells
        self.saved_nearest_neighbor_file = saved_nearest_neighbor_file

    def build_index(self, dataloader: AbsDataloader = None, faiss_fast: bool = False,
                    device: str = None,
                    batch_size: int = 64, data_type: str = "train", ):
        data_items = get_doc_use_tuple([])
        data_items = [DataItem(text=item[0], label=item[1],query_str=item[2],task_id=item[3],query_lebal=item[4],user_id=item[5],query_id=item[6],serp_id=item[7],order=item[8])  for item in data_items]
        data_instances=data_items
        self.text_md5_to_label = {encode_md5hash(item.text): item.label for item in data_instances}
        self.text_md5_to_query_str = {encode_md5hash(item.text): item.query_str for item in data_instances}
        self.text_md5_to_task_id = {encode_md5hash(item.text): item.task_id for item in data_instances}
        self.text_md5_to_query_lebal = {encode_md5hash(item.text): item.query_lebal for item in data_instances}
        self.text_md5_to_user_id = {encode_md5hash(item.text): item.user_id for item in data_instances}
        sentences = [item.text for item in data_instances]
        self.saved_nearest_neighbor_file= None
        if self.saved_nearest_neighbor_file is None:
            self._build_index(sentences, faiss_fast=faiss_fast, device=device, batch_size=batch_size)
        else:
            self._build_index_from_saved_file(self.saved_nearest_neighbor_file)

    def _build_index_from_saved_file(self, saved_nearest_neighbor_file: str, ):
        query_and_nearest_neighbors = load_jsonl(saved_nearest_neighbor_file)
        self.index = {}
        for item in query_and_nearest_neighbors:
            self.index[item["query_text"]] = item

    def _build_index(self, sentences: List[str],
                     faiss_fast: bool = False,
                     device: str = None,
                     batch_size: int = 64):
        embeddings = self.encode(sentences, batch_size=batch_size, normalize_to_unit=True, )
        self.index = {"sentences": sentences}
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        if faiss_fast:
            index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences)),
                                       faiss.METRIC_INNER_PRODUCT)
        else:
            index = quantizer

        if (self.device == "cuda" and device != "cpu") or device == "cuda":
            if hasattr(faiss, "StandardGpuResources"):
                self.logger.info("Use GPU-version faiss")
                res = faiss.StandardGpuResources()
                res.setTempMemory(20 * 1024 * 1024 * 1024)
                index = faiss.index_cpu_to_gpu(res, 0, index)
        if faiss_fast:
            index.train(embeddings.astype(np.float32))
        index.add(embeddings.astype(np.float32))
        index.nprobe = min(self.num_cells_in_search, len(sentences))
        self.is_faiss_index = True
        self.index["index"] = index

    def search(self, queries: Union[str, List[str]],
               threshold: float = 0.0,
               top_k: int = 1) -> Tuple[str, np.float32]:
        """
        Returned-results:
        """
        if self.saved_nearest_neighbor_file is not None:
            if isinstance(queries, str):
                print("=" * 10)
                if queries in self.index.keys():
                    results = self.index[queries]["nearest_neighbors"]
                    clip_results = [(item["text"], item["score"]) for item in results if item["score"] >= threshold]
                    return clip_results[:top_k]
                else:
                    raise ValueError(queries)
            elif isinstance(queries, list):
                results = []
                for query in queries:
                    result = self.index[query]["nearest_neighbors"]
                    clip_result = [(item["text"], item["score"]) for item in result if item["score"] >= threshold]
                    results.append(clip_result)
                return results
            else:
                raise ValueError
        else:
            query_vecs = self.encode(queries, normalize_to_unit=True, keepdim=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)

            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results

            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])

    def search_nearest_neighbors_and_save_to_file(self, save_file_path: str, dataloader: AbsDataloader = None,
                                                  candidate_type: str = "train", query_type: str = "test",
                                                  search_threshold: float = 0.2,
                                                  top_k: int = 1, ranking_model: str = "roberta-large",
                                                  ranking: str = "h2l"):
        check_file_and_mkdir_for_save(save_file_path, file_suffix=".jsonl")
        self.build_index(dataloader, data_type=candidate_type)
        query_data_lst = dataloader.load_data_files(query_type)
        writer_f = open(save_file_path, "w")
        for query_data in tqdm(query_data_lst, desc="nearest-neighbors"):
            query_text = query_data.text
            query_label = query_data.label
            nearest_neighbors = self.search(query_text, threshold=search_threshold, top_k=top_k)
            nearest_neighbors_label = [self.text_md5_to_label[encode_md5hash(item[0])] for item in
                                       nearest_neighbors]
            nearest_neighbors_text = [item[0] for item in nearest_neighbors]
            nearest_neighbors_score = [float(item[1]) for item in nearest_neighbors]
            nearest_neighbors_lst = [{"text": text, "label": label, "score": score} for text, label, score
                                     in zip(nearest_neighbors_text, nearest_neighbors_label, nearest_neighbors_score)]
            saved_result_item = {"query_text": query_text, "query_label": query_label,
                                 "nearest_neighbors": nearest_neighbors_lst,
                                 "search_threshold": search_threshold, "top_k": top_k,
                                 "ranking_model": ranking_model, "ranking": ranking}
            writer_f.write(f"{json.dumps(saved_result_item)}\n")
        writer_f.close()

class SimCSERetriever(AbsDataRetriever):
    def __init__(self, mlm_name_or_path: str, max_len: int = 128, **kwargs):
        super(SimCSERetriever, self).__init__(**kwargs)
        self.tokenizer = Tokenizer(mlm_name_or_path, max_len=max_len, pad_to_max_length=True,
                                   return_offsets_mapping=False)
        self.saved_nearest_neighbor_file = None
        if self.saved_nearest_neighbor_file is None:
            self.model = self.init_encoder(mlm_name_or_path, )
            self.model.to(self.device)

    @classmethod
    def init_encoder(cls, mlm_name_or_path: str, ):
        encoder_config = AutoConfig.from_pretrained(mlm_name_or_path)
        encoder_model = AutoModel.from_pretrained(mlm_name_or_path, config=encoder_config)
        encoder_model.eval()
        return encoder_model

    def encode(self, sentence: Union[str, List[str]], normalize_to_unit: bool = True,
               keepdim: bool = False, batch_size: int = 64, ) -> np.array:
        if isinstance(sentence, str):
            sentence = [sentence]

        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in range(total_batch):
                inputs = self.tokenizer.tokenize_input_batch(
                    sentence[batch_id * batch_size:(batch_id + 1) * batch_size], )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        embeddings = torch.cat(embedding_list, 0)
        embeddings = embeddings.numpy()
        return embeddings
