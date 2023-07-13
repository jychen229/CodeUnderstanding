#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from tqdm import *
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import pickle

from utils import *
from models import PositionalEncoding, GAT, Node2Vec, TranslationModel, GCN, SAGE
from torch_geometric.nn import to_hetero

import argparse
parser = argparse.ArgumentParser()

#----------GNN hype-parameters-------------
parser.add_argument('--gnn', help='SAGE, GAT', default='SAGE')
parser.add_argument('--gnn_layers', default=2)
parser.add_argument('--gnn_hidden_dim', default=256)

#----------Translator--------------------
parser.add_argument('--use_gnn', help='Add graph module', default=True)
parser.add_argument('--encoder_layers', default=3)
parser.add_argument('--decoder_layers', default=3)
parser.add_argument('--num_heads', default = 4)
parser.add_argument('--dim_feedforward', default = 512)
parser.add_argument('--hidden_dim', default=256)
parser.add_argument('--alpha', default=0.5)
parser.add_argument('--batch_size', default=128)

with open('./dataset_process/train_output.txt', 'r') as f:
    tgt_train = f.readlines() 
    
with open('graph.pkl', 'rb') as f:
    graph = pickle.load(f)
    
args = parser.parse_args()

    
train, valid, test = pre_process()
Train_data = TranslationDataset(train,tgt_train,src_tokenizer, tgt_tokenizer,80,80)
train_loader = DataLoader(Train_data, batch_size= args.batch_size , shuffle=True, collate_fn=collate_fn)

alpha = args.alpha
if args.use_gnn==False:
    alpha = 0
    
    
if args.gnn == 'GAT':
    GNN = GAT(hidden_channels=args.gnn_hidden_dim, out_channels=args.hidden_dim,num_layers =  args.gnn_layers)
else:
    GNN = SAGE(args.gnn_hidden_dim, args.hidden_dim, args.gnn_layers)

GNN = to_hetero(GNN, graph.metadata(), aggr='sum')

model = TranslationModel(args.hidden_dim, 
                         args.num_heads, 
                         args.encoder_layers,
                         args.decoder_layers,
                         args.dim_feedforward,
                         src_tokenizer.vocab, 
                         tgt_tokenizer.vocab,
                         alpha)

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criteria = TranslationLoss()

step = 0

step = int('model_10000.pt'.replace("model_", "").replace(".pt", ""))
epochs = 10
model.train()
for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for index, data in enumerate(train_loader):
        src, tgt, tgt_y = data
        #src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
        optimizer.zero_grad()
        out1 = GNN(graph.x_dict, graph.edge_index_dict)
        gnn_out = Node2Vec(out1,80,src_tokenizer,args.hidden_dim)
        emb = gnn_out(src)
        out = model(src, tgt,emb)
        out = model.predictor(out)

        loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) 
        loss.backward()
        optimizer.step()

        loop.set_description("Epoch {}/{}".format(epoch, epochs))
        loop.set_postfix(loss=loss.item())
        loop.update(1)

        step += 1

        del src
        del tgt
        del tgt_y

        if step != 0 and step % 500 == 0:
            torch.save(model, f"./model_{step}.pt")
