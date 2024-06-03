#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


# model = 'gin'
# hidden = [64]

para = [
        ('ogbn-products'      ,  'gcn', 64) ,   
        ('ogbn-products'      ,  'gcn', 128) , 
        ('ogbn-products'      ,  'gcn', 256) ,
        ('ogbn-products'      ,  'gcn', 512) ,
]

print('*********Running experiments in fig 15c begin*********\n')
for data, model, hidden_dim in para:
    command = "python ../train_fastgl.py --dataset_name  {} \
                --model {} --num_hidden {} --log_path logs_fig15c/".format(data, model, hidden_dim)		
    os.system(command)
print('*********Running experiments in fig 15c complete!*********\n')