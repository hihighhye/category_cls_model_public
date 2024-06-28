import sys
import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pytz import timezone

import torch
# import torch.nn as nn
import torch.nn.functional as F
# from torchtext import data

import pandas as pd
import numpy as np
import copy
import os
import re
import ast
# import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
from tqdm import tqdm
import traceback
import copy
import requests, json

# bertshared-kor-base
# 학습데이터 추상 클래스 Dataset, DataLoader
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, BertForSequenceClassification, AlbertForSequenceClassification, BertTokenizerFast, EncoderDecoderModel, AutoModelForQuestionAnswering, TrainingArguments, Trainer

from transformers import DistilBertTokenizer, DistilBertModel # 토크나이저
from tokenizers import BertWordPieceTokenizer

# 아래 2개의 패키지 설치!!!!필수 !!!! (pymysql, sshtunnel)
from sqlalchemy import create_engine
import pymysql
from sshtunnel import SSHTunnelForwarder
from lgcns_connection_setting import *


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_len', type=int, default=100)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config

# 라벤슈타인 유사도 (edit distance)
def levenshteinDistanceDP(token1, token2):
    """
    token1 -> item_master (subbrand)
    token2 -> title token
    """
    # spell = SpellChecker()
    # token2 = spell.correction(token2)
    # spell = Speller(lang='en')
    # token2 = spell(token2)
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
    a, b, c = 0, 0, 0
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]

                if t1 and t2 and token1[t1 - 1] == token2[t2 - 2] and token1[t1 - 2] == token2[t2 - 1]:
                    distances[t1 - 1][t2 - 1] = min(distances[t1 - 1][t2 - 1], distances[t1 - 3][t2 - 3] + 0)
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

                if t1 and t2 and token1[t1 - 1] == token2[t2 - 2] and token1[t1-2] == token2[t2 - 1]:
                    distances[t1 - 1][t2 - 1] = min(distances[t1 - 1][t2 - 1], distances[t1 - 3][t2 - 3] + 1)
    dist = distances[-1][-1]
    sim = (len(token1) - dist) / len(token1)
    return sim

def ngram(s, num):
    res = []
    slen = len(s) - num + 1
    for i in range(slen):
        ss = s[i:i + num]
        res.append(ss)
    return res

def diff_ngram(token_a, token_b, num):
    a = ngram(token_a, num)
    b = ngram(token_b, num)

    r = []
    cnt = 0
    for i in a:
        for j in b:
            if i == j:
                cnt += 1
                r.append(i)

    return cnt / len(a)

def prepare_testset(df):
    texts, labels = [], [] # 라벨, 텍스트 리스트 선언
        
    for i in range(len(df)):
        text = '' 
        if type(df.loc[i, 'option_name']) != '':
            text = df.loc[i, 'option_name']
        text += '\t' + df.loc[i, 'title']

        label = df.loc[i, 'search_keyword']

        texts += [text]
        labels += [label]

    return texts, labels

def classify_data(config, df, cls_model_fn):
    saved_data = torch.load(
        cls_model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    cls_train_config = saved_data['config']
    cls_bert_best = saved_data['bert']
    cls_index_to_label = saved_data['classes']
    cls_tokenizer = saved_data['tokenizer']

    contexts, _ = prepare_testset(df)
    result_df = pd.DataFrame()

    with torch.no_grad():
        model_loader = AlbertForSequenceClassification if cls_train_config.use_albert else BertForSequenceClassification
        model = model_loader.from_pretrained(
            cls_train_config.pretrained_model_name,
            num_labels=len(cls_index_to_label)
        )

        model.resize_token_embeddings(len(cls_tokenizer))
        model.load_state_dict(cls_bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in tqdm(range(0, len(contexts), config.batch_size)):
            mini_batch = cls_tokenizer(
                contexts[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            # Take feed-forward
            y_hat = F.softmax(model(x, attention_mask=mask).logits, dim=-1)

            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(1)
        # |indice| = (len(lines), top_k)
    
        temp_dict = {'cls_pred': [ cls_index_to_label[int(ind[0])] for ind in indice ], 'probs': [float(pb[0]) for pb in probs]}
        pred_df = pd.DataFrame.from_dict(temp_dict)
        # print(pred_df)

        m_result_df = pd.concat([df, pred_df], axis=1).reset_index(drop=True)
        result_df = pd.concat([result_df, m_result_df], axis=0).reset_index(drop=True)
    return result_df

def check_result(df):
    for i, row in tqdm(enumerate(df.itertuples())):
        answer_set = set(row.group_category.split(','))
        if row.cls_pred in answer_set:
            df.loc[i, 'check'] = 1
    return df

if __name__ == '__main__':
    start_time = time.time()

    config = define_argparser()

    cls_model_fn = 'lg_11cat_cls_m1'

    print('#' * 100)
    print('used model: {}'.format(cls_model_fn))
    print('#' * 100)


    test_df = pd.read_csv('whole_cat_test_dataset.csv', encoding='utf-8-sig')
    test_df = test_df.fillna('')

    result_df = pd.DataFrame()
    for i in tqdm(range(0, len(test_df), 10000)):
        mini_df = test_df[i:i+10000].reset_index(drop=True)

        m_result_df = classify_data(config, mini_df, cls_model_fn)
        result_df = pd.concat([m_result_df, result_df], axis=0)

    result_df = result_df.reset_index(drop=True)
    final_df = check_result(result_df)
    final_df.to_csv(f'{cls_model_fn}_result.csv', encoding='utf-8-sig', index=False)

    final_df = pd.read_csv(f'{cls_model_fn}_result.csv', encoding='utf-8-sig')
    final_df['check'] = final_df['check'].fillna(0)
    correct_cnt = final_df['check'].sum()
    print(f'correct count: {correct_cnt}')
    print(f'leftover: {len(final_df) - correct_cnt}')

    

    end_time = time.time()
    print("실행 시간 : {}".format(timedelta(seconds=(end_time-start_time))))









