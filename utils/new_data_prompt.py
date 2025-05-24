
from utils.new_data_loader import AbsDataloader
import json
import os
from typing import Dict, List, Union
from utils.new_data_retriever import SimCSERetriever
from utils.new_data_utils import Detokenizer,  encode_md5hash

from utils.new_file_utils import  DataItem
class Prompt(object):


    __slots__ = ["model_backbone", "prompt_strategy", "instance_num", "instance_strategy", "gradient_update"]

    def __init__(self, key_value_params: Dict = None):
        if key_value_params is not None:
            for key in self.__slots__:
                if key in key_value_params.keys():
                    self.__setattr__(key, key_value_params[key])
                    key_value_params.pop(key)
                else:
                    self.__setattr__(key, None)
            if len(key_value_params) != 0:
                raise ValueError(key_value_params)



    def map_predicted_verbalizer_to_label(self):
        raise NotImplementedError

    def _get_config(self):
        config_pairs = {}
        for slot_key in self.__slots__:
            if slot_key in ["tokenizer", "data_retriever", "detokenizer"]:
                slot_value = None
            elif slot_key in ["dataloader"]:
                slot_value = str(self.__getattribute__(slot_key))
            else:
                try:
                    slot_value = self.__getattribute__(slot_key)
                except:
                    slot_value = None
            config_pairs[slot_key] = slot_value
        return config_pairs

    def __str__(self):
        """return the string."""
        config_data = self._get_config()
        return json.dumps(config_data, indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json_file(cls, config_path: str):
        """load config from json assets."""
        with open(config_path, "r", encoding="utf-8") as f:
            config_items = json.load(f)
        filtered_configs = {key: value for key, value in config_items.items() if key in cls.__slots__}
        return cls(filtered_configs)

    def save_to_json(self, save_path: str):
        """save config to file."""
        config_pairs = self._get_config()
        if os.path.exists(save_path):
            raise FileExistsError(f"{save_path}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_pairs, f, sort_keys=True, indent=2, ensure_ascii=False)
        print(f"SAVE CONFIG TO {save_path}")


class GPT3FewShotSamplingPrompt(Prompt):
    __slots__ = Prompt.__slots__ + ["task_description", "delimiter", "demonstration_pattern", "verbalizer",
                                    "feasible_verbalizer",
                                    "assemble_demonstration_strategy", "max_prompt_len", "inverse_verbalizer",
                                    "detokenizer", "verbalizer_position_idx", "demonstration_subtask_description",
                                    "assemble_demonstration_pattern", "data_retriever", "data_retriever_candidate_dir",
                                    "retriever_name_or_path",
                                    "retriever_ckpt_path", "file_saved_retriever_results", "demonstration_ranking",
                                    "non_verbalizer", "dataloader", "max_instance_len", "max_explain_len",
                                    "model_generate_max_len", "demonstration_subtask_description_pos", "prompt_suffix","background"]

    def __init__(self, key_value_params: Dict = None):
        super(GPT3FewShotSamplingPrompt, self).__init__(key_value_params)
        import yaml
        config_path = './config/config.yaml'
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if config.get('data_name')=="KDD":
            self.background = {
            }
        else:
            self.background = {
            }

        self.dataloader = AbsDataloader()
        self.instance_num = 3
        self.max_instance_len = 200
        self.assemble_demonstration_strategy = "model_generate"

        self.max_explain_len = 100
        self.instance_strategy = "no-simcse-nearest-neighbor"
        self.demonstration_subtask_description_pos = 0
        self.delimiter="\n\n"
        self.max_prompt_len = 1024
        self.demonstration_subtask_description=""
        self.feasible_verbalizer = {label_id: label_token.lower() for label_id, label_token in
                                        self.verbalizer.items()}
        self.inverse_verbalizer = {}
        for label_symbol, label_word in self.feasible_verbalizer.items():
            if isinstance(label_word, list):
                for token in label_word:
                    assert not any(element.isupper() for element in token)
                    self.inverse_verbalizer[token] = label_symbol
            elif isinstance(label_word, str):
                assert not any(element.isupper() for element in label_word)
                self.inverse_verbalizer[label_word] = label_symbol
            else:
                raise ValueError(self.inverse_verbalizer)
        self.detokenizer = Detokenizer()

        if self.instance_strategy == "simcse-nearest-neighbor":
            data_retriever_loader = self.dataloader

            self.data_retriever = SimCSERetriever(mlm_name_or_path="../data/models/bert-base-chinese", max_len=256,
                                                  saved_nearest_neighbor_file=None)
            self.data_retriever.build_index(data_retriever_loader)


    def _clip_text_by_space_len(self, input_text: str, max_space_len: int = 200) -> str:
        input_token = input_text.split(" ")
        if len(input_token) <= max_space_len:
            return input_text

        input_token_clipped = input_token[:max_space_len]
        input_text_clip = " ".join(input_token_clipped)
        return input_text_clip




