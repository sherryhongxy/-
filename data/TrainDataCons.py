# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:12:57 2024

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
import requests
import time
from tqdm import tqdm


def random_swap(text, source, cnt=1):
    if source=='webqa':
        text_list = list(text)
    else:
        text_list = text.split(' ')
    for i in range(cnt):
        index_a = random.randint(0,len(text_list)-1)
        index_b = index_a+1
        if index_b<len(text_list) and text_list[index_a].isdigit()==False and text_list[index_b].isdigit()==False:
            text_list[index_a], text_list[index_b] = text_list[index_b], text_list[index_a]
    if source=='webqa':
        return ''.join(text_list)
    return ' '.join(text_list)
        


def swap(context, source, ans):
    global cnt
    context_list = re.split(r"(，|。|；|！|？|：|,|\.|;|!|\?|:)", context)
    index_list = [index for index in range(len(context_list)) if len(context_list[index])>=5]
    if_mask = random.random() 
    if if_mask<1 and len(index_list)>0:
        swap_index = sample(index_list, 1)[0]
        if ans in context_list[swap_index]:
            cnt+=1
        context_list[swap_index] = random_swap(context_list[swap_index], source, 2)
        context = ''.join(context_list)
    return context



cnt = 0
def mask(context, source, ans, prop=0.1):
    global cnt
    context_list = re.split(r"(，|。|；|！|？|：|,|\.|;|!|\?|:)", context)
    index_list = [index for index in range(len(context_list)) if len(context_list[index])>1] 

    ans_index = []
    for index in index_list:
        if ans in context_list[index]:
            ans_index.append(index)
    res_index_list = [index for index in index_list if index not in ans_index]
    if_mask = random.random() 
    if len(index_list)>0:
        if len(ans_index)>0 and (len(res_index_list)==0 or if_mask<=prop):
            mask_span_index = ans_index[0]
            cnt+=1
        else:
            mask_span_index = sample(res_index_list, 1)[0]
        del context_list[mask_span_index]
        if mask_span_index<len(context_list) and len(context_list[mask_span_index])<=1:
            del context_list[mask_span_index]
        context = ''.join(context_list)
    return context



