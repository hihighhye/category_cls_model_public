# 상품 카테고리 분류 모델

[beomi/kcbert-base](https://github.com/Beomi/KcBERT-finetune) pretrained model을 100만건의 실제 전자제품 상품 데이터로 finetuned한 한국어 텍스트 상품 카테고리 분류 모델

<br>

> **Objective** : 전자제품 상품 판매 페이지 타이틀과 옵션 텍스트를 기반으로 해당 상품을 아래 11개 카테고리 중 하나로 분류.

<br>
 
 | Category | 상세 포함 항목 |
 |----------|-------------|
 | 냉장고 | 김치냉장고, 냉장고 등 |
 | 노트북 | 노트북, 게이밍노트북, 태블릿 PC 등 |
 | 세탁기+건조기+의류관리기 | 세탁기, 건조기, 의류관리기, 스타일러, 워시타워 등 |
 | TV | 스텐드형 TV, 벽걸이형 TV, TV+모니터 등 |
 | 청소기 | 청소기, 로봇청소기, 무선청소기 등 |
 | 에어컨 | 에어컨, 벽걸이에어컨, 스탠드에어컨, 창문형에어컨, 멀티형에어컨, 이동식에어컨, 에어컨+TV 등 |
 | 모니터 | 사무용 모니터, 게이밍 모니터 등 |
 | 공기청정기 | 공기청정기 등 |
 | 전기레인지 | 전기레인지, 인덕션, 휴대용 전기레인지 등 |
 | 식기세척기 | 식기세척기, 업소용 식기세척기, 식기세척기+전기레인지(인덕션) 등 |
 | 제습기 | 제습기, 공기청정제습기, 미니제습기, 대용량 제습기, 화장실제습기 등 |


<br>


## How To Use

### Requirements
* torch==1.12.1+cu116
* transformers==4.31.0

<br>

```
import sys
import argparse
import time
from datetime import date, datetime, timedelta, timezone
from pytz import timezone

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

# bertshared-kor-base
# 학습데이터 추상 클래스 Dataset, DataLoader
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator, BertForSequenceClassification, AlbertForSequenceClassification, BertTokenizerFast, EncoderDecoderModel, AutoModelForQuestionAnswering, TrainingArguments, Trainer

from transformers import DistilBertTokenizer, DistilBertModel # 토크나이저
from tokenizers import BertWordPieceTokenizer



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

def prepare_testset(df):
    texts, labels = [], [] # 라벨, 텍스트 리스트 선언
        
    for i in range(len(df)):
        text = '' 
        if type(df.loc[i, 'option_text']) != '':
            text = df.loc[i, 'option_text']
        text += '\t' + df.loc[i, 'title']

        label = df.loc[i, 'category']

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

        probs, indice = y_hats.cpu().topk(1)
    
        temp_dict = {'cls_pred': [ cls_index_to_label[int(ind[0])] for ind in indice ], 'probs': [float(pb[0]) for pb in probs]}
        pred_df = pd.DataFrame.from_dict(temp_dict)

        m_result_df = pd.concat([df, pred_df], axis=1).reset_index(drop=True)
        result_df = pd.concat([result_df, m_result_df], axis=0).reset_index(drop=True)
    return result_df


if __name__ == '__main__':
    config = define_argparser()

    cls_model_fn = 'lg_11cat_cls_m1'

    test_df = pd.read_csv('test_dataset_sample.csv', encoding='utf-8-sig')
    test_df = test_df.fillna('')

    result_df = classify_data(config, test_df, cls_model_fn)

```

<br>


## Train Data & Preprocessing
 
* Input : 1,000,000건의 전자제품 상품 판매 페이지의 타이틀과 옵션 텍스트 <br>
  *띄어쓰기/특수기호 등에 대한 text-cleansing은 따로 진행하지 하지 않고 raw text 그대로 사용. <br>
    > <em>Input Format</em> :  ``` [옵션 텍스트] + '\t' + [타이틀 텍스트] ``` ( prepare_testset() 함수 참조. ) <br>
         E.g.
        >> Title text : [티몬] LG 오브제컬렉션 공기청정기 AS354NS4A+무빙휠 무료배송 신세계 <br>
            Option text : AS354NS4AM <br>
            => 'AS354NS4AM' + '\t' + '[티몬] LG 오브제컬렉션 공기청정기 AS354NS4A+무빙휠 무료배송 신세계'

* Labels : 각 상품의 카테고리
    
    > <em>Labels</em> :  
    ['냉장고', '노트북', '세탁기+건조기+의류관리기', 'TV', '청소기', '에어컨', '모니터', '공기청정기', '전기레인지', '식기세척기', '제습기']
  
<br>

* 각 카테고리별 학습데이터 건수:
  
    | Category | 건수 |
    |----------|-------------:|
    | 냉장고 | 298,492 |
    | 노트북 | 197,700 |
    | 세탁기+건조기+의류관리기 | 105,670 |
    | TV | 94,835 |
    | 청소기 | 79,821 |
    | 에어컨 | 74,948 |
    | 모니터 | 51,241 |
    | 공기청정기 | 33,626 |
    | 전기레인지 | 27,838 |
    | 식기세척기 | 22,404 |
    | 제습기 | 13,425 |
    | 총 계 | 1,000,000 |


 <br>

## Used Tokenizer

* `BertWordPieceTokenizer` from the pretrained model.
* 카테고리를 표현하는 단어(e.g. 냉장고 - 와인냉장고, 김치냉장고 등) 및 국내에서 판매되는 전자제품의 주요 브랜드명, 주요 모델명을 토큰으로 추가 등록. <br><br>
*  `카테고리별 브랜드명 및 모델명 추가 토큰 수`
    | Category | 브랜드명 토큰 | 모델명 토큰 |
    |----------|-------------:|-----------:|
    | 냉장고 | 117 | 6,586 |
    | 노트북 | 21 | 4,292 |
    | 세탁기+건조기+의류관리기 | 71 | 2,634 |
    | TV | 102 | 3,221 |
    | 청소기 | 279 | 1,711 |
    | 에어컨 | 70 | 2,119 |
    | 모니터 | 102 | 2,807 |
    | 공기청정기 | 254 | 1,273 |
    | 전기레인지 | 181 | 1,234 |
    | 식기세척기 | 71 | 890 |
    | 제습기 | 106 | 366 |

    *Total extra tokens(unique() 기준) : **925 brands & 27,104 items** 

<br>


## Performance

* 총 47,293건의 테스트 데이터셋으로 테스트한 결과.
* 정답과 다르게 분류된 건 중 11개 카테고리 외의 상품은 어느 카테고리로 분류하든 True Negative로 인정.
* 정답과 다르게 분류된 건 중 추가 토큰으로 등록한 모델명 외의 상품은 마찬가지로 어느 카테고리로 분류하든 True Negative로 인정.

<br>

        Accuracy : 96.64% (45,703 / 47,293) | Precision : 1.0 | Recall : 0.99
        F1 Score : 0.97 (weighted) / 0.87 (macro)

<br>


## Citation

* kcBERT ( Used pretrained model )
```
@inproceedings{lee2020kcbert,
  title={KcBERT: Korean Comments BERT},
  author={Lee, Junbum},
  booktitle={Proceedings of the 32nd Annual Conference on Human and Cognitive Language Technology},
  pages={437--440},
  year={2020}
}
```


## Reference
* https://www.markdownguide.org/extended-syntax/#highlight



