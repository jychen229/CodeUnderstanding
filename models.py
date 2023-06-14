#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, Linear, to_hetero

import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1, 7个单词，单词维度为128
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
    

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x.relu()
    
    
class Node2Vec(nn.Module):
    def __init__(self, graph_model, max_len, tokenzier, d_model):
        super(Node2Vec,self).__init__()
        self.embedding = graph_model
        self.max_len = max_len
        self.decoder = tokenzier
        self.pre_embedding = nn.Embedding(3, d_model, padding_idx=2)
        self.d_model = d_model
        
    def decode(self,input):
        tmp = [self.decoder.decode(i[1:]).split(' [SEP]')[0].split(' ') for i in input]
        ls = []
        #print(tmp)
        for i in tmp:
            ls.append([int(j) for j in i])
        return  ls #[list(map(int, i)) for i in tmp]
        
    def forward(self,x):
        '''
        x is the encoder nodes [batch size, max len]
        '''
        # decode x
        nodes = self.decode(x)

        # tackle the [CLS] [SEP] [PAD]
        cls = self.pre_embedding(torch.tensor(0))
        sep = self.pre_embedding(torch.tensor(1))
        #pad = self.pre_embedding(torch.tensor(2))
        emb = torch.zeros((x.shape[0],x.shape[1],self.d_model))
        
        for i in range(x.shape[0]):
            tmp = torch.zeros((self.max_len, self.d_model))
            tmp[0] = cls
            tmp[1:1+len(nodes[i]),:] = self.embedding['Node'][nodes[i],:]
            if len(nodes[i])< self.max_len-1 :
                tmp[1+len(nodes[i])] = sep
            emb[i] =  tmp 
        return emb
    

class TranslationModel(nn.Module):

    def __init__(self, d_model, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()

        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=80)
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt, emb):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        src = emb
        tgt = self.tgt_embedding(tgt)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 0
