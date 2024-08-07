# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:56:13 2023

@author: dell
"""


import json
import numpy as np
import pandas as pd
import os
import random
from random import sample, shuffle
import jsonlines
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import Any, Dict, List
import platform
import subprocess
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig
import time 
from tqdm import tqdm
import warnings
import logging
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.DEBUG)


def post_web_search_res(se_list, topn=3):
    if se_list[0]=='\n':
       se_list = se_list[1:]
    se_res = [re.sub(r"网页.*?\u2002·\u2002", "", item) for item in se_list[:topn]]
    for i in range(len(se_res)):
        if se_res[i][-1]=="\n":
            se_res[i] = se_res[i][:-1]
    # print(se_res)
    se_res = [str(i+1)+'. '+se_res[i] for i in range(len(se_res))]
    return ' '.join(se_res)



class TestInfer():
    def __init__(self, model_name, data_path, model_type):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            trust_remote_code=True
        )        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        ) 
        self.model_type = model_type
        if model_type in ['bc']:
           self.model.generation_config = GenerationConfig.from_pretrained(model_name)
           
        with open(data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
            


    def chat(self, text):
        response = ''
        if self.model_type in ['bc']:
            messages=[{"role": "user", "content": text}]
            response = self.model.chat(self.tokenizer, messages)
        elif self.model_type in ['llama']:
            device = "cuda:0"
            generation_config = GenerationConfig(
                temperature=0.2,
                top_k=5,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.1,
                max_new_tokens=400
            )
            template = "[INST]{}[/INST]"
            text = template.format(text)
            inputs = self.tokenizer(text, return_tensors="pt")
            generation_output = self.model.generate(input_ids = inputs["input_ids"].to(device), 
                                               attention_mask = inputs['attention_mask'].to(device),
                                               eos_token_id=self.tokenizer.eos_token_id,
                                               pad_token_id=self.tokenizer.eos_token_id,  #Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
                                               generation_config = generation_config)
            response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
            response = response[len(text):].strip()
            # print("response:", response)
        elif self.model_type in ['glm']:
            response, history = self.model.chat(self.tokenizer, text, history=[])  
        return response
            
        
        
    def infer(self, IF_PRINT = 0):
        prompt_kbqa_en = """As a question answering robot, your task is to answer questions based on given knowledge. \
        Note that: \
        Triples are in the format of 'subject ||| predicate ||| object' and is delimited by semicolon: {}; \
        The question is: {} \
        """
        
        prompt_kbqa_zh = """作为一个问答机器人，你的任务是根据给定的知识回答问题。\
        请注意：\
        三元组的格式为'subject ||| predicate ||| object'，并用分号分隔：{}' \
        问题为：{} \
        """
        
        prompt_mrc_en = """As a question answering robot, your task is to answer questions based on given knowledge. \
        Addition context: {}; \
        The question is: {} \
        """
        
        prompt_mrc_zh = """作为一个问答机器人，你的任务是根据给定的知识回答问题。\
        请注意：\
        文本为：{} \
        问题为：{} \
        """
        
        #多源-infer
        prompt_en = """As a question answering robot, your task is to answer questions based on knowledge from different sources. \
        Note that: \
        Triples are in the format of 'subject ||| predicate ||| object' and is delimited by semicolon: {}; \
        Addition context:  {} \
        The question is: {} \
        """
        
        
        prompt_zh= """作为一个问答机器人，你的任务是根据不同来源的知识回答问题。\
        请注意： \
        三元组的格式为'subject ||| predicate ||| object'，并用分号分隔：{}' \
        文本为：{} \
        问题为：{} \
        """
        
        mrc_source = ['cmrc', 'ms', 'squad', 'webqa']
        kbqa_source = ['NLPCC-MH', 'CKBQA', 'KQA Pro']
        
        for i in tqdm(range(len(self.test_data))):
            data = self.test_data[i]
            data['output'] = data['answer']
            #单源
            if data['source'] in mrc_source:
                if data['lang']=='en':
                    data['instruction_single'] = prompt_mrc_en.format(data['text'], data['question']) 
                else:
                    data['instruction_single'] = prompt_mrc_zh.format(data['text'], data['question']) 
            else:
                if data['lang']=='en':
                    data['instruction_single'] = prompt_kbqa_en.format(data['triple'], data['question']) 
                else:
                    data['instruction_single'] = prompt_kbqa_zh.format(data['triple'], data['question']) 
            
           
            response = self.chat(data['instruction_single'])
            if IF_PRINT==1:
                print('instruction_single:', data['instruction_single'])
                print('answer_single:', response)
            data['answer_single'] = response
            

            if data['lang']=='en':
                data['instruction_align'] = prompt_en.format('; '.join(data['triple'].split(' &&&& ')), data['text'], data['question']) 
            else:
                data['instruction_align'] = prompt_zh.format('; '.join(data['triple'].split(' &&&& ')), data['text'], data['question']) 
            
            response = self.chat(data['instruction_align'])
            if IF_PRINT==1:
                print('instruction_align:', data['instruction_align'])
                print('answer_align:', response)
            data['answer_align'] = response
            


        for i in tqdm(range(len(self.test_data))):
            data = self.test_data[i]
            data['output'] = data['answer']
            if data['source'] in mrc_source: 
                data['concerned'] = "; ".join(data['concerned'].split(' &&&& ')[:10]) 
                if data['lang']=='en':
                    data['instruction_concerned'] = prompt_en.format(data['concerned'], data['text'], data['question'])
                else:
                    data['instruction_concerned'] = prompt_zh.format(data['concerned'], data['text'], data['question'])
            else:
                data['concerned'] = post_web_search_res(data['concerned'], 3)
                if data['lang']=='en':
                    data['instruction_concerned'] = prompt_en.format(data['triple'], data['concerned'], data['question']) 
                else:
                    data['instruction_concerned'] = prompt_zh.format(data['triple'], data['concerned'], data['question']) 

            response = self.chat(data['instruction_concerned'])
            if IF_PRINT==1:
                print('instruction_concerned:', data['instruction_concerned'])
                print('answer_concerned:', response) 
            data['answer_concerned'] = response
            
        
        for i in tqdm(range(len(self.test_data))):
            data = self.test_data[i]
            if data['index_states']!='success':
                continue
            if data['source'] in mrc_source: 
                if data['lang']=='en':
                    data['instruction_conflict'] = prompt_en.format("; ".join(data['fake'].split(' &&&& ')) , data['text'], data['question'])
                else:
                    data['instruction_conflict'] = prompt_zh.format("; ".join(data['fake'].split(' &&&& ')), data['text'], data['question'])
            else:     
                if data['lang']=='en':
                    data['instruction_conflict'] = prompt_en.format(data['triple'], data['fake'], data['question']) 
                else:
                    data['instruction_conflict'] = prompt_zh.format(data['triple'], data['fake'], data['question']) 
        
            response = self.chat(data['instruction_conflict'])
            if IF_PRINT==1:
                print('instruction_conflict:', data['instruction_conflict'])
                print('answer_conflict:', response)
            data['answer_conflict'] = response


        for i in tqdm(range(len(self.test_data))):
            data = self.test_data[i]
            try:
                data['delete']
            except:
                continue
            if data['source'] in mrc_source:
                if data['lang']=='en':
                    data['instruction_reject'] = prompt_mrc_en.format(data['delete'], data['question']) 
                else:
                    data['instruction_reject'] = prompt_mrc_zh.format(data['delete'], data['question']) 
            else:
                if data['lang']=='en':
                    data['instruction_reject'] = prompt_kbqa_en.format(data['delete'], data['question']) 
                else:
                    data['instruction_reject'] = prompt_kbqa_zh.format(data['delete'], data['question']) 
        
            response = self.chat(data['instruction_reject'])
            if IF_PRINT==1:
                print('instruction_reject:', data['instruction_reject'])
                print('answer_reject:', response)
            data['answer_reject'] = response






