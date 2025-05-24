import numpy as np
import logging
from collections import defaultdict
logging.basicConfig(level=logging.ERROR)
from yacs.config import CfgNode
from utils.new_data_loader import AbsDataloader
from agents.data import Data
from utils.new_file_utils import load_jsonl, get_num_lines, check_file_and_mkdir_for_save, DataItem, save_jsonl, save_json,DataItem_new,DataItem_sigir
from utils.new_data_prompt import GPT3FewShotSamplingPrompt
import os
from agents.recagent import Agent

class GPT3TextCLS(object):
    def __init__(self, config: CfgNode, log_interval: int = 50):
        self.config = config

        self.prompt = GPT3FewShotSamplingPrompt()
        self.dataloader = AbsDataloader()
        self.log_interval = log_interval
        os.environ["OPENAI_API_KEY"] = self.config["api_keys"][0]
        self.data = Data(self.config)
        self.model = Agent()
        import pandas as pd
        if self.config["data_name"] == "KDD":
            file_path = './data/KDD19zong_query_id.xlsx'
            self.df = pd.read_excel(file_path)

    def doc_to_query_jsonl(self, query_save_path: str, step1_prompt_data_path: str, stage: str):
        saved_result_path = os.path.join(self.config["save_log_dir"], f"query{int(stage)}_input.jsonl")
        check_file_and_mkdir_for_save(saved_result_path, file_suffix=".jsonl")
        data_item_lst = load_jsonl(step1_prompt_data_path)

        batches = defaultdict(list)
        for item in data_item_lst:
            batches[str(item['query_id'])].append(item)

        all_batch_prompt_text = []

        def query_level_feature(item, df):
            search_id = item["query_id"]
            row = df.iloc[search_id]
            assert int(item["query_lebal"]) == int(row['Q_SAT'])

            if row.empty:
                return ""
            session_end_text = "这是整个搜索会话的最后一个查询。" if row['isSessionEnd'] else "这不是整个搜索会话的最后一个查询。"
            prompt = (
                f"[{int(item['order']) + 1}]"
                f"{item['text']}。该文档是搜索者第{int(item['order']) + 1}个点击进入阅读的文档，该文档在文档列表中的排名是第{int(item['serp_id']) + 1}名，"
                f"点击的文档的平均排名是 {round(row['AvgClickRank'], 1)}位。"
                f"点击的最深的文档是第 {int(round(row['ClickDepth']))}位。"
                f"信息搜索者对该文档的阅读时间为{np.ceil(item['dwell_time'] / 1000)}秒。"
                f"而搜索者在此查询中阅读每个文档的平均时间是 {round(row['AvgContent'] / 1000, 1)}秒，"
            )
            return prompt

        def format_document_new(item):
            if len(item['clicked_ranks_list']) == 1:
                return (
                        f"[{int(item['order']) + 1}]" + item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "搜索者只点击了这一个文档就完成了搜索。" +
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。"
                )
            else:
                return (
                        f"[{int(item['order']) + 1}]" + item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "而搜索者在此查询下总共点击了" + str(int(item['total_clicks_number'])) + "个文档，" +
                        "点击的所有文档的排名是" + str([x + 1 for x in item['clicked_ranks_list']]) + "。" +
                        "最大的点击深度是" + str(item['max_clicked_rank'] + 1) + "。" +
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。" +
                        "而搜索者在此查询上的所有文档的平均阅读时长是" + str(
                    np.ceil(item['avg_dwell_time'] / 1000)) + "秒。"
                )

        def generate_batch_prompt(batch_data_item, example_str="", stage=""):
            batch_prompt_text = [
                {"role": "system",
                 "content": "你是信息检索领域的智能助手，能够根据信息搜索者的信息需求和查询对文档的有用性进行排序。"}
            ]
            if example_str:
                batch_prompt_text.append({"role": "user", "content": example_str})

            # 添加问题描述
            batch_prompt_text.append({
                "role": "user",
                "content": f"现在的信息需求和查询是: {batch_data_item[0]['prompt_text']}。我会提供给你 {len(batch_data_item)} 个文档，每个文档都使用一个[数字]进行标识，请根据它们的实用性进行排序。"
            })
            batch_prompt_text.append({'role': 'assistant', 'content': '好的，请提供文档。'})

            # 添加文档描述
            for i, item in enumerate(batch_data_item):
                if self.config["data_name"] == "KDD":
                    doc_prompt = query_level_feature(item, self.df)
                else:
                    doc_prompt = format_document_new(item)
                batch_prompt_text.append({'role': 'user', 'content': doc_prompt})
                batch_prompt_text.append({'role': 'assistant', 'content': '已接收该文档。'})

            batch_prompt_text.append({
                "role": "user",
                "content": "现在请对这些文档进行排序，按照它们的实用性从高到低排列，输出格式应为 [i]>[j]>...>[k]。请严格按照这个格式输出，不要输出任何其他的字符或者汉字。"
            })
            return batch_prompt_text
        return saved_result_path

    def doc_to_query_jsonl_4(self, query_save_path: str, step1_prompt_data_path: str, stage: str):
        saved_result_path = os.path.join(self.config["save_log_dir"], f"query{int(stage)}_input_4.jsonl")
        check_file_and_mkdir_for_save(saved_result_path, file_suffix=".jsonl")
        data_item_lst = load_jsonl(step1_prompt_data_path)
        batches = defaultdict(list)
        for item in data_item_lst:
            batches[str(item['query_id'])].append(item)

        all_batch_prompt_text = []

        def query_level_feature(item, df):
            search_id = item["query_id"]
            row = df.iloc[search_id]
            assert int(item["query_lebal"]) == int(row['Q_SAT'])
            if row.empty:
                return ""
            session_end_text = "这是整个搜索会话的最后一个查询。" if row['isSessionEnd'] else "这不是整个搜索会话的最后一个查询。"
            prompt = (
                f"[{int(item['order']) + 1}]"
                f"{item['text']}。该文档是搜索者第{int(item['order']) + 1}个点击进入阅读的文档，该文档在文档列表中的排名是第{int(item['serp_id']) + 1}名，"
                f"点击的文档的平均排名是 {round(row['AvgClickRank'], 1)}位。"
                f"点击的最深的文档是第 {int(round(row['ClickDepth']))}位。"
                f"信息搜索者对该文档的阅读时间为{np.ceil(item['dwell_time'] / 1000)}秒。"
                f"而搜索者在此查询中阅读每个文档的平均时间是 {round(row['AvgContent'] / 1000, 1)}秒，"
            )
            return prompt

        def format_document_new(item):
            if len(item['clicked_ranks_list']) == 1:
                return (
                        f"[{int(item['order']) + 1}]" + item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "搜索者只点击了这一个文档就完成了搜索。" +
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。"
                )
            else:
                return (
                        f"[{int(item['order']) + 1}]" + item['text'] +
                        "。该文档是搜索者在该查询中第" + str(int(item['order']) + 1) + "个点击进入阅读的文档。" +
                        "该文档在文档列表中的排名是第" + str(int(item['serp_id']) + 1) + "名。" +
                        "而搜索者在此查询下总共点击了" + str(int(item['total_clicks_number'])) + "个文档，" +
                        "点击的所有文档的排名是" + str([x + 1 for x in item['clicked_ranks_list']]) + "。" +
                        "最大的点击深度是" + str(item['max_clicked_rank'] + 1) + "。" +
                        "搜索者对该文档的阅读时间为" + str(np.ceil(item['dwell_time'] / 1000)) + "秒。" +
                        "而搜索者在此查询上的所有文档的平均阅读时长是" + str(
                    np.ceil(item['avg_dwell_time'] / 1000)) + "秒。"
                )

        def generate_batch_prompt(batch_data_item, example_str="", stage=""):
            batch_prompt_text = [
                    {"role": "system",
                     "content": "你是分数归一化的助手，能够根据现在的排序得分列表给出归一化为四级或者五级分数的分数列表。"}
                ]
            batch_prompt_text.append({
                "role": "user",
                "content": f"现在的信息需求和查询是: {batch_data_item[0]['prompt_text']}。我会提供给你 {len(batch_data_item)} 个文档，每个文档都使用一个[数字]进行标识，文档和排序得分列表的顺序是一一对应的。"
            })
            score_list=[]

            for i, item in enumerate(batch_data_item):
                if self.config["data_name"] == "KDD":
                    doc_prompt = query_level_feature(item, self.df)
                else:
                    doc_prompt = format_document_new(item)
                batch_prompt_text.append({'role': 'user', 'content': doc_prompt})
                score_list.append(item["pred_label"])
                if self.config["data_name"] == "KDD":
                    batch_prompt_text.append({
                        "role": "user",
                        "content": "这些文档的有用性排序得分分别是："+str(score_list)+"，请考虑文档的有用性，帮我将这个排序列表归一化为 4，3，2，1的四级分数。输出格式为一个长度相同的列表[]，例如[4,3,2,1]。请严格按照这个格式输出，不要输出任何其他的字符或者汉字。"
                    })
                else:
                    batch_prompt_text.append({
                        "role": "user",
                        "content": "这些文档的有用性排序得分分别是：" + str(score_list) + "，请考虑文档的有用性，帮我将这个排序列表归一化为 5，4，3，2，1的五级分数。输出格式为一个长度相同的列表[]，例如[5,3,2,1]。请严格按照这个格式输出，不要输出任何其他的字符或者汉字。"
                    })
            return batch_prompt_text
        return saved_result_path

