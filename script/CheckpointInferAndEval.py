
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation.utils import GenerationConfig
import time 
from tqdm import tqdm
import warnings
import logging
from typing import List
#from rouge_chinese import Rouge
#from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.DEBUG)
from peft import PeftModel
import shutil
#import jieba
import argparse
from transformers import AutoTokenizer, AutoModel
from TestInfer import TestInfer
from EvalMetrics import EvalMetrics

def merge_lora_to_base_model(model_name_or_path, save_path, output_dir, type='bc'):
    # model_name_or_path = '../../llama_models/Baichuan2-13B-Chat/Baichuan2-13B-Chat'
    # adapter_name_or_path = 'output/bc2-13b-qlora-sft-44w-to-30p-zh-prompts-bz-chat/final'
    # save_path = 'checkpoint/bc2-13b-qlora-sft-44w-to-30p-zh-prompts-bz-chat'
    
    model_name_or_path = model_name_or_path
    adapter_name_or_path = output_dir
    print('model_name_or_path:', model_name_or_path)
    print('adapter_name_or_path:', adapter_name_or_path)
    print('save_path:', save_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    #bc2还需要把py文件加进去
    if type=='bc':
        dstfile = ['configuration_baichuan.py', 'generation_utils.py', 'modeling_baichuan.py',
                   'quantizer.py', 'tokenization_baichuan.py']
        for file in dstfile:
            shutil.copyfile('checkpoint/bc2-config/'+file, save_path+'/'+file) 
    elif type=='glm':
        dstfile = ['configuration_chatglm.py', 'modeling_chatglm.py',
                   'quantization.py', 'tokenization_chatglm.py']
        for file in dstfile:
            shutil.copyfile('checkpoint/glm3-config/'+file, save_path+'/'+file) 



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=int, default=1)
    parser.add_argument("--model_name_or_path", type=str, default="../../llama_models/Baichuan2-13B-Chat/Baichuan2-13B-Chat") #../../llama_models/Baichuan2-7B-Chat
    parser.add_argument("--output_dir", type=str, default="output/bc2-13b-sft-4w-mrc-single-source-swap-mask-all")
    parser.add_argument("--final_model_path", type=str, default='checkpoint/tmp')
    parser.add_argument("--start_step", type=int, default=500)
    parser.add_argument("--end_step", type=int, default=5000)
    parser.add_argument("--intv_step", type=int, default=500)
    parser.add_argument("--type", type=str, default='bc')
    parser.add_argument("--datapath", type=str, default='paper/_dev_test_webqa_squad_1016.json')
    args = parser.parse_args()
    return args



args = parse_args()
print(args)

if args.lora==1:
    final_model_path = args.final_model_path
    output_dir = args.datapath.split('.')[0]+ '_' + args.output_dir.split('/')[-1] + '.json'
else:
    final_model_path = args.model_name_or_path
    output_dir = args.datapath.split('.')[0]+ '_' + args.model_name_or_path.split('/')[-1] + '.json'
    
print('final_model_path:', final_model_path)
print('output_dir:', output_dir)   


# final_model_path = args.final_model_path
# output_dir = args.output_dir


dirs_list = []
for root, dirs, files in os.walk(args.output_dir, topdown=False):
    for name in dirs:
        if name=='final':
            dirs_list.append(os.path.join(root, name))   
        if 'checkpoint' in name:
            if int(name.split('-')[-1])>=args.start_step and int(name.split('-')[-1])<=args.end_step and int(name.split('-')[-1])%args.intv_step==0:
                dirs_list.append(os.path.join(root, name))
 
if dirs_list==[]:
    dirs_list.append(args.output_dir)
print(dirs_list) 


ans_list = ['answer_single', 'answer_reject', 'answer_reject_conf', 'answer_align', 'answer_align_conf', 
            'answer_concerned', 'answer_concerned_conf', 'answer_conflict', 'answer_conflict_conf']


# try:
#     shutil.rmtree(args.final_model_path)
# except:
#     pass
    
res_df = pd.DataFrame()
for path in dirs_list:
    if args.lora==1:
        print(path)
        merge_lora_to_base_model(args.model_name_or_path, args.final_model_path, path, args.type)

    TestData = TestInfer(final_model_path, data_path=args.datapath, model_type=args.type)
    TestData.infer()
    EvalFunc = EvalMetrics()
    
    save_file = path+'/infer.json'
    with open(save_file,"w",encoding = 'utf-8' ) as dump_f:
        json.dump(TestData.test_data, dump_f,ensure_ascii=False, indent = 2)    
    
    

    ans = []
    for key in ans_list:
        print(key)
        ans.append(EvalFunc.no_ans_cnt(TestData.test_data, key))
        ans.append(EvalFunc.conflict_cnt(TestData.test_data, key))
        ans.append(EvalFunc.same_cnt(TestData.test_data, key, 'answer'))
        ans.append(EvalFunc.ans_in(TestData.test_data, key, 'answer'))
        ans.append(EvalFunc.avg_lens(TestData.test_data, key))
        ans.append(EvalFunc.same_cnt(TestData.test_data, key, 'answer_fake'))
    res_df[path] = ans
    # res_df.to_csv(args.output_dir+'/res_df.csv', index=False)
    

    acc, rouge1, rouge2, rougel, bleu4, recall, lens = [], [], [], [], [], [], []
    res_dict = EvalFunc.compute_metrics(TestData.test_data, ans_key='answer_single')
    acc.append(res_dict['accuracy'])
    rouge1.append(res_dict['rouge-1'])
    rouge2.append(res_dict['rouge-2'])
    rougel.append(res_dict['rouge-l'])
    bleu4.append(res_dict['bleu-4'])
    recall.append(res_dict['recall'])
    lens.append(res_dict['lens'])

    check_df = pd.DataFrame({'model':dirs_list[:len(acc)], 'acc':acc, 'rouge1':rouge1, 'rouge2':rouge2, 'rougel':rougel, 'bleu4':bleu4, 'recall':recall, 'lens':lens})

    
    res_df.to_csv(args.output_dir+'/res_df.csv', index=False)
    check_df.to_csv(args.output_dir+'/check_df.csv', index=False)













