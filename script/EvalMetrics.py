# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:10:03 2024

@author: dell
"""



import json
import pandas as pd
import jieba
import os
import requests
import jsonlines
import time
from tqdm import tqdm
import copy
import random
import numpy as np
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def is_chinese(strings):
    for _char in strings:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

class EvalMetrics():
    
    def __init(self):
        self.reject_key_words = ['无法提供', '需要更多', '提供更多', '提供有关','无法确定', '无答案', '无法回答', '抱歉', 
                            'context is not sufficient', 'information is not sufficient', 'I need to know', 'try to offer', 
                            'context is insufficient', 'information is insufficient', 'Unfortunately', 'Sorry', 'sorry',
                            'no answer', 'No answer', 'No Answer']
    
    def same_cnt(self, dataset, ans_key2='answer', ans_key1='answer'):
        same = 0
        for item in dataset:
            try:
                if item[ans_key1].lower()==item[ans_key2].lower():
                    same+=1
            except:
                pass
        print('same with {} is: {}'.format(ans_key1, same))  
        return same
    
    
    def ans_in(self, dataset, ans_key2='answer', ans_key1='answer'):
        same = 0
        for item in dataset:
            try:
                if item[ans_key1].lower() in item[ans_key2].lower():
                    same+=1
            except:
                pass
        print('in the {} is: {}'.format(ans_key1, same))  
        return same
    
    
    def avg_lens(self, dataset, ans_key='answer'):
        total_lens = 0
        for i in range(len(dataset)):
            try:
                if dataset[i]['lang']=='en':
                    total_lens += len(dataset[i][ans_key].split(' '))
                else:
                    total_lens += len(dataset[i][ans_key])
            except:
                pass 
        print('avg_lens:', total_lens/len(dataset))
        return round(total_lens/len(dataset),2)
    
    
    def no_ans_cnt(self, dataset, ans_key='answer'):
        tmp = []
        dist = {}
        cnt = 0
        for item in dataset:
            flag = 0
            try:
                for key in self.reject_key_words:
                    if key in item[ans_key]:
                        flag=1
                        break
                if flag==1 or item['question'][:-1] in item[ans_key]: 
                    tmp.append(item)
                    cnt+=1
                    if item['source'] in dist.keys():
                        dist[item['source']]+=1
                    else:
                        dist[item['source']]=1
            except:
                pass
        print('reject:', cnt)   
        return cnt


    def compute_metrics(self, dataset, ans_key='answer'):
        
        decoded_preds, decoded_labels, langs = [], [], []
        for item in dataset:
            decoded_preds.append(item[ans_key])
            decoded_labels.append(item['answer'])
            langs.append(item['lang'])
        
        
        score_dict = {"accuracy": [], "rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": [], "recall": []}
        
        pred_lens, label_lens = 0.0, 0.0
        for pred, label, lang in zip(decoded_preds, decoded_labels, langs):
    
            if lang=='zh':
                hypothesis = list(jieba.cut(pred))
                reference = list(jieba.cut(label))
                pred_lens += len(pred)
                label_lens += len(label)
            else:
                #大小写转换
                hypothesis = pred.lower().split(' ')
                reference = label.lower().split(' ')
                pred_lens += len(hypothesis)
                label_lens += len(reference)
            
            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]
        
            for k, v in result.items():
                score_dict[k].append(v["f"])
            
            if lang=='zh':
                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            else:
                bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(bleu_score)
    
            score_dict["accuracy"].append(float(len(label) != 0 and pred[:len(label)] == label))
            
            recall = 0 
            for token in list(set(hypothesis)):
                if token in list(set(reference)):
                    recall+=1
            score_dict["recall"].append(recall/len(list(set(reference))))
            
        
        res_dict = {k: round(float(np.mean(v)),4) for k, v in score_dict.items()}    
        res_dict['samples'] = len(decoded_labels)
        res_dict['lens'] = round(pred_lens/res_dict['samples'],4)

        return res_dict
