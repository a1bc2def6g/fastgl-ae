#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


# model = 'gin'
# hidden = [64]

para_gcn = [
        ('reddit'             ,  'gcn') ,   
        ('ogbn-products'      ,  'gcn') , 
        ('ogbn-papers100M'    ,  'gcn') ,
        ('mag'                ,  'gcn') ,
        ('igb-large'          ,  'gcn') ,   
]

para_gin = [
        ('reddit'             ,  'gin') ,   
        ('ogbn-products'      ,  'gin') , 
        ('ogbn-papers100M'    ,  'gin') ,
        ('mag'                ,  'gin') ,
        ('igb-large'          ,  'gin') ,   
]

para_gat = [
        ('reddit'             ,  'gat') ,   
        ('ogbn-products'      ,  'gat') , 
        ('ogbn-papers100M'    ,  'gat') ,
        ('mag'                ,  'gat') ,
        ('igb-large'          ,  'gat') ,   
]

print('*********Running experiments on GCN begin*********\n')
for data, model in para_gcn:
    command = "python ../train_dgl.py --dataset_name  {} \
                --model {}".format(data, model)		
    os.system(command)
print('*********Running experiments on GCN complete!*********\n')

print('*********Running experiments on GIN begin*********\n')
for data, model in para_gin:
    command = "python ../train_dgl.py --dataset_name  {} \
                --model {}".format(data, model)		
    os.system(command)
print('*********Running experiments on GIN complete!*********\n')

print('*********Running experiments on GAT begin*********\n')
for data, model in para_gat:
    command = "python ../train_dgl.py --dataset_name  {} \
                --model {}".format(data, model)		
    os.system(command)
print('*********Running experiments on GAT complete!*********\n')
