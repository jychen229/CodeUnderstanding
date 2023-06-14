#!/usr/bin/env python
# coding: utf-8


from torch_geometric.data import HeteroData
import numpy as np
import pickle
import torch
import json
import re


# Nodes [operates]  -> sentence

print('start building hetero graph...')

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


def content2feature(nodx_index):   # node2vec
    fea = [0 for i in range(24)]
    cont = content[nodx_index].split()
    for i in range(min(24,len(cont))):
        fea[i] = list(operate).index(cont[i])
    return fea

operate = set('A')
for i in content.keys():
    cont = set(content[i].split())
    operate.update(cont)


sent_fea = [] #sent_features
for i in train:
    fea = []
    for j in i:
        fea.append(content2feature(j))
    sent_fea.append(np.mean(fea,axis=0).tolist())
for i in valid:
    fea = []
    for j in i:
        fea.append(content2feature(j))
    sent_fea.append(np.mean(fea,axis=0).tolist())
for i in test:
    fea = []
    for j in i:
        fea.append(content2feature(j))
    sent_fea.append(np.mean(fea,axis=0).tolist())

node = []
lis = content.keys()
for i in range(23984):
    if str(i) in lis:
        node.append(content2feature(str(i)))
    else:
        node.append([0 for j in range(24)])

link = []
for i in links:
    link.append([int(i[0]),int(i[1])])

all_nodes_in = train+valid+test
node_in_s = []
for i in range(len(all_nodes_in)):
    for j in all_nodes_in[i]:
        node_in_s.append([int(j),i])

data = HeteroData()
data['Node'].x = torch.Tensor(node)
data['Sentence'].x = torch.tensor(sent_fea)
data['Node', 'connect', 'Node'].edge_index = torch.tensor(link).T
data['Node', 'in', 'Sentence'].edge_index = torch.tensor(node_in_s).T

with open('graph.pkl', 'wb') as f:
    pickle.dump(data, f)
print('finished!')