#!/usr/bin/env python
# coding: utf-8

import torch
from transformers import BertTokenizer
import pickle

from utils import *
from models import PositionalEncoding, GAT, Node2Vec, TranslationModel
from torch_geometric.nn import to_hetero


src_tokenizer = BertTokenizer("./srcvocab.txt")

with open('graph.pkl', 'rb') as f:
    graph = pickle.load(f)
    
node2vec = torch.load('./node2vec11500.pt')
gnn = torch.load('./GNN_11500.pt')
model = torch.load('./model_11500.pt')

train, valid, test = pre_process()

def translate(src: list):
    # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
    src =torch.tensor([src])
    # 首次tgt为<bos>
    tgt = torch.tensor([[2]])
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(80):
        # 进行transformer计算
        
        out1 = gnn(graph.x_dict, graph.edge_index_dict)
        emb = node2vec(src)
        out = model(src, tgt,emb)
        
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == 3:
            break
    return tgt


s1 = src_tokenizer.encode(test[0], padding='max_length', truncation= True, max_length = 80)
s2 = translate(s1)
res = tgt_tokenizer.decode(s2[0].data.detach())

