# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:12:57 2024

@author: dell
"""
'''data construction'''

from elasticsearch import Elasticsearch
import json
from elasticsearch.helpers import bulk
import pandas as pd
import jieba
import os
import requests
import jsonlines
import copy
import numpy as np
import time
import jieba.analyse
import numpy as np
from random import sample, shuffle
import re
from tqdm import tqdm


def talk_to_gpt(question, temperature=1, model="gpt-3.5-turbo", use="test", cache=False, n=1, max_try=2):
    url = ""

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "messages": [
            {'role': 'assistant', 'content': 'You are a helpful assistant'},
            {"role": "user", "content": question}
        ],
        "temperature": temperature,
        "use": use,
        "cache": False,
        "model": model,
        "n":n
    }
    
    try_num = 0
    while True and try_num<max_try:
        try_num +=1
        try:
            response = requests.post(url, headers=headers, json=data)
            break
        except Exception as e:
            response = str(e)
            print(response)
            if ("limit" in str(response)):
                print('****************wait for 30 seconds**************')
                time.sleep(30)
            else:
                break
    
    try:
        return response.json()['response']
    except:
        return response

def Context2Triple(context, lang='zh'):
    sample_zh = """Context:他与杜甫并称为“大李杜”，（李商隐与杜牧并称为“小李杜”）。
    Response: [
           {
               "subject": "杜甫",
               "predicate": "被称为",
               "object": "大李杜"
           },
           {
              "subject": "李商隐",
              "predicate": "被称为",
              "object": "小李杜"
           },
           {
               "subject": "杜牧",
               "predicate": "被称为",
               "object": "小李杜"
           }
           ]

    """

    sample_en = """Context: We all like to think of yogurt as a healthy breakfast, but as it turns out, many of the most popular yogurt brands have more sugar than a junk food you'd never consider eating. The American Heart Association recommends that men eat no more than 36 grams of sugar per day, and women no more than 20. One Twinkie makes a big dent in that recommended daily max, packing 19 grams of the sweet stuff, Time reported. Many of the top-selling yogurts have even more.
    Response: [            
            {
               "subject": "yogurt",
               "predicate": "is",
               "object": "healthy breakfast"
           },
           {
              "subject": "many of the most popular yogurt brands",
              "predicate": "have more sugar than",
              "object": "a junk food you'd never consider eating"
           },
           {
               "subject": "The American Heart Association",
               "predicate": "recommends",
               "object": "men eat no more than 36 grams of sugar per day, and women no more than 20."
           },
           {
               "subject": "Twinkie",
               "predicate": "sugar content",
               "object": "19 grams of the sweet stuff"
           },
           {
               "subject": "Many of the top-selling yogurts",
               "predicate": "sugar content",
               "object": "even more than Twinkie"
           }
           ]

    """    
    
    if lang=='zh':
        sample = sample_zh
    else:
        sample = sample_en
        
    template = f"""
            You are an AI language model that has been trained to extract triplets containing subjects, attributes, and objects from a given context. \
            Given context:{context} \
            Your task is to generate List composed by JSON objects with the keys "subject", "predicate", and "object". \
            A given Example: {sample} \
            Return in JSON format.
           """
    # print(template)
    news = []


    try:
        response = talk_to_gpt(template)
        print(response)
        news = json.loads(response)

    except Exception as e:

        print(str(e))

    return news


def Triples2Str(DictOrList):
    triple_list = []
    if isinstance(DictOrList, list):
        for i in DictOrList:
            try:
                if len(i['subject'])>0 and len(i['predicate'])>0 and len(i['object'])>0:
                    triple_list += [str(i['subject']) + " ||| " + str(i['predicate']) + " ||| " + str(i['object'])]
            except Exception as e:
                print(e)
                continue
    triple_str = '; '.join(triple_list)
    return triple_str



def get_response_from_google_serper(question: str, language: str):
    url = "https://google.serper.dev/search"

    if language == 'en':
        payload = json.dumps({
            "q": question,
            "gl": "cn"
        })
    elif language == 'zh':
        payload = json.dumps({
            "q": question,
            "gl": "cn",
            "hl": "zh-cn"
        })
    else:
        raise RuntimeError('Serper google api language error: ' + language)

    headers = {
        'X-API-KEY': '',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    results = json.loads(response.text)
    return results



def describe(dict_list):
    sample_list = []
    sources = {}
    for data in dict_list:
        if data['source'] not in sources.keys():
            sources[data['source']] = 1
            print(data)
        else:
            sources[data['source']] +=1
    print(sources)


def research_triples_subject(subject, size=30, lang='zh'):
    es_host = ""
    es_username = "elastic"
    es_password = ""
    index_name = 'kg_max'
    
    es = Elasticsearch(
        [es_host],
        http_auth=(es_username, es_password),
        verify_certs=False
    )
    
    if lang=='zh':
        query = {
            "query": {
                "match": {"subject": subject}
            },
            "sort": {
                "_score": {
                    "order": "desc"
                }
            }
        }
    else:
        query = {
            "query": {
                "bool": {
                    "must": [{"match": {"subject": subject}}],
                    "should": [
                        {"match": {"kg_source": "KQA_Pro"}},
                        {"match": {"kg_source": "MetaQA"}}
                        ],
                    "minimum_should_match": 1
                    }
                }
            }


    triples = []
    res = es.search(index=index_name, body=query)
    # print(res)
    response=res['hits']['hits']
    # print(response)
    
    for d in response:
        if len(triples)<size:
            triples.append(str(d['_source']['subject'])+' ||| '+str(d['_source']['relation'])+' ||| '+str(d['_source']['object']))
        else:
            break

    return ' &&&& '.join(triples)



def delete_answer(item):
    if item['source'] in ['cmrc', 'webqa']:
        joint = '。'
        context = item['text']
    elif item['source'] in ['ms', 'squad']:
        context = item['text']
        joint = '.'
    else:
        context = item['triple']
        joint = '; '
    context_list = context.split(joint)
    flag = 0
    if item['answer'] not in context or len(context_list)<=1: 
        pass
    else:
        answer = item['answer'].split(joint)[0]
        for i in range(len(context_list)-1,-1,-1):
            if answer in context_list[i]:
                flag=1
                del context_list[i]    
        if flag==1:         
            item['delete'] = joint.join(context_list)
        if len(item['delete'])<1:
            del item['delete']
            flag=0
    return flag, item['source']



def gen_fake_ans(item):
    prompt = """
    According to the question and answer, give one incorrect answer that is different from given answer but can answer the question. \
    question: {} \
    answer: {} \
    Please return in Json format: {{\"Output\": new answer}}
    Example 1:
    question: 小苹果是谁写的？
    answer: 筷子兄弟
    return: {{\"Output\": 凤凰传奇}} 
    Example 2:
    question: What is the relationship between Nord and Somme?
    answer: shares border with
    return: {{\"Output\": do not shares a border}}
    """

    output_gt = item['answer']

    text = item['text']
    triple = item['triple']
    question = item['question']
    
    index_states = 'success'
    if 'fake' in item.keys():
        continue
    if  output_gt in text and output_gt in triple:
        instruction = prompt.format(question, output_gt)
        response = talk_to_gpt(instruction)
        try:
            output_fake = json.loads(response)['Output']
        except:
            try:
                output_fake = json.loads(response.split('\n')[-1].split('：')[1])['Output'] 
            except:
                output_fake = response  
                index_states = 'gpt_parse_error'
                fail_index.append(i)
        context_fake = triple.replace(output_gt, str(output_fake))
    else: 
        index_states = 'not_in_gen'
        output_fake = ''
        context_fake = ''
        fail_index.append(i)
    item['fake_answer'] = str(output_fake)
    item['fake'] = str(context_fake)
    item['index_states'] = index_states



def keyword_extract_search(item):
    keywords = jieba.analyse.extract_tags(item['question'])
    keywords = [keyword for keyword in keywords if keyword.isdigit()==False and '哪' not in keyword and '.' not in keyword and '__' not in keyword]
    item['keywords'] = '_#_'.join(keywords)
    flag = 0
    for keyword in keywords: #search
        res = get_response_from_google_serper(keyword, 'zh')
        for res_i in res['organic']:
            if item['answer'] not in res_i['snippet']:
                flag = 1
                item['delete'] = res_i['snippet']
                break
        if flag==1:
            break
    if flag==0:
        fail_index.append(i) 
    



