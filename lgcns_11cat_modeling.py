import argparse
import random
import gc
gc.collect() 

import torch
print(torch.cuda.empty_cache())

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, AlbertForSequenceClassification, BertTokenizerFast, AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim
import pandas as pd
import re
import os

from simple_ntc.bert_trainer import BertTrainer as Trainer
from simple_ntc.bert_dataset import TextClassificationDataset, TextClassificationCollator
from lgcns_connection_setting import *




def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--cls_model_name', required=True)
    # Recommended model list:
    # - kykim/bert-kor-base
    # - kykim/albert-kor-base
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')
    p.add_argument('--use_albert', action='store_true')
    
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=3)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    # If you want to use RAdam, I recommend to use LR=1e-4.
    # Also, you can set warmup_ratio=0.
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--valid_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config

def sample_trainset(df, limit=0):
    df = df.sample(frac=1).reset_index(drop=True)
    if limit != 0:
        df = df[:limit]
    return df

def prepare_trainset(df):
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

def get_loaders(df, tokenizer, valid_ratio=.2):    # , n_limit=300, um_limit=1000
    # Get list of labels and list of texts.
    # hashcodes, texts, labels, qi_pcs, qi_bds, cat_classes = read_text(fn, n_limit, um_limit)
    texts, labels = prepare_trainset(df)

    # Generate label to index map.
    unique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        TextClassificationDataset(texts[:idx], labels[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )
    valid_loader = DataLoader(
        TextClassificationDataset(texts[idx:], labels[idx:]),
        batch_size=config.batch_size,
        collate_fn=TextClassificationCollator(tokenizer, config.max_length),
    )

    return train_loader, valid_loader, index_to_label

def get_optimizer(model, config):
    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    return optimizer

def get_extra_tokens_list(df):
    return list(set(df['top_node_title']) | set(df['middle_node_title']) | set(df['bottom_node_title']))

def main(config):
    # Get pretrained tokenizer.
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    automatching_server = AutoDb(AUTOMATCHING_SERVER)

    schema_id = '756'
    config_id = '849'

    auto_db_name = 'lg_cns_22cat'

    hash_table_name = f'hash_data_po'
    final_md_table_name = f'mass_final_md_data_po'
    md_table_name = f'mass_md_data_po'
    bon_table_name = f'mass_md_lgcns_mi_ecsm_{schema_id}_{config_id}_to' # f'mass_md_lgcns_mi_ecsm_{schema_id}_{config_id}_to'

    item_master = automatching_server.import_item_master()
    extra_tokens_list = get_extra_tokens_list(item_master)
    print('#'*30, 'Extra tokens are ready!', '#'*30)
    added_token_num = tokenizer.add_tokens(extra_tokens_list)
    print('#'*30, 'Tokens are added successfully!', '#'*30)

    final_df = automatching_server.import_data(auto_db_name, final_md_table_name, conditions='refine_code!=0 and search_keyword!=\'전체\'')
    print('#'*30, f'final_md is loaded!:{len(final_df)}', '#'*30)

    train_df = sample_trainset(final_df, 1000000)
    train_df.to_csv(f'{config.cls_model_name}_train_dataset.csv', encoding='utf-8-sig', index=False)

    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders(
        train_df,
        tokenizer,
        # n_limit=100,
        # um_limit=100,
        valid_ratio=config.valid_ratio

    )

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
    )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    optimizer = get_optimizer(model, config)

    # By default, model returns a hidden representation before softmax func.
    # Thus, we need to use CrossEntropyLoss, which combines LogSoftmax and NLLLoss.
    crit = nn.CrossEntropyLoss()
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    # Start train.
    trainer = Trainer(config)

    model = trainer.train(
        model,
        crit,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
    )
    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.cls_model_name)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
