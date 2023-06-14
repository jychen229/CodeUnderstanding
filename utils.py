#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.nn.functional import  log_softmax
import os
import math
import numpy as np
import torch.nn.functional as F
import json
import re

tgt_tokenizer = BertTokenizer("./tgtvocab.txt")
src_tokenizer = BertTokenizer("./srcvocab.txt")

def pre_process():
    with open('./dataset_process/train_input.txt', 'r') as f:
        input_lines = f.readlines()
    with open('./dataset_process/validate_input.txt', 'r') as f:
        input_lines2 = f.readlines()
    with open('./dataset_process/test_input.txt', 'r') as f:
        input_lines3 = f.readlines()

    input_train = input_lines
    input_val = input_lines2
    input_test = input_lines3

    # dataset pre_process :
    input_set = set()
    links = set()
    train_nodes = []
    valid_nodes = []
    test_nodes = []
    for i in range(len(input_train)):
        line = input_train[i]
        nodes = re.findall(r'N(\d+)\[label=\"(.*?)\"\]', line)
        link =  re.findall(r'N(\d+) -> N(\d+)',line)
        links.update(set(link))
        train_nodes.append(list(link))
        nodes_set = set(nodes)
        input_set.update(nodes_set)

    for i in range(len(input_val)):
        line = input_val[i]
        nodes = re.findall(r'N(\d+)\[label=\"(.*?)\"\]', line)
        nodes_set = set(nodes)
        link =  re.findall(r'N(\d+) -> N(\d+)',line)
        links.update(set(link))
        valid_nodes.append(list(link))
        input_set.update(nodes_set)

    for i in range(len(input_test)):
        line = input_test[i]
        nodes = re.findall(r'N(\d+)\[label=\"(.*?)\"\]', line)
        nodes_set = set(nodes)
        link =  re.findall(r'N(\d+) -> N(\d+)',line)
        links.update(set(link))
        test_nodes.append(list(link))
        input_set.update(nodes_set)


    content = {}      # node content
    for i in input_set:
        content[str(i[0])]=i[1]

    train, valid, test = [], [], []   # 句子包含的node
    for i in train_nodes:
        node = set()
        for j in i:
            node.update(j)
        train.append(list(node))
    for i in valid_nodes:
        node = set()
        for j in i:
            node.update(j)
        valid.append(list(node))
    for i in test_nodes:
        node = set()
        for j in i:
            node.update(j)
        test.append(list(node))
    train[7186] = ['7299']
    train[7187] = ['7299']
    
    return train, valid, test


class TranslationDataset(Dataset):

    def __init__(self, src_list, tgt_list, src_tokenizer, tgt_tokenizer, src_max_len, tgt_max_len):
        self.src_tokens = self.load_tokens(src_list, src_tokenizer, max_len = src_max_len)
        self.tgt_tokens = self.load_tokens(tgt_list, tgt_tokenizer, max_len = tgt_max_len)
        
    def __getitem__(self, index):
        return self.src_tokens[index], self.tgt_tokens[index]

    def __len__(self):
        return len(self.src_tokens)
    
    def load_tokens(self, input_list, tokenizer, max_len):
        tokens_list = []  #labels 
        for i in input_list:
            tokens_list.append(tokenizer.encode(i, padding='max_length', truncation= True, max_length = max_len))
        return tokens_list
    
    
def collate_fn(batch):
    tgt = []
    src = []
    tgt_y = []
    for (_src, _tgt) in batch:
        src.append(_src)
        tgt_y.append(_tgt)
        tgt.append(_tgt)
    tgt_y = torch.tensor(tgt_y)[:,1:]
    src = torch.tensor(src)
    tgt = torch.tensor(tgt)[:,:-1]
    tgt[tgt == 3] = 0
    return src, tgt, tgt_y


class TranslationLoss(nn.Module):
    def __init__(self):
        super(TranslationLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = 0

    def forward(self, x, target):

        x = log_softmax(x, dim=-1)
        true_dist = torch.zeros(x.size())
        true_dist.scatter_(1, target.data.unsqueeze(1), 1)
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())