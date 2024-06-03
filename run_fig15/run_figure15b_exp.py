#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


# model = 'gin'
# hidden = [64]

para = [
        ('ogbn-products'      ,  'gcn', 4000) ,   
        ('ogbn-products'      ,  'gcn', 6000) , 
        ('ogbn-products'      ,  'gcn', 8000) ,
        ('ogbn-products'      ,  'gcn', 12000) ,
]

print('*********Running experiments in fig 15b begin*********\n')
for data, model, bs in para:
    command = "python ../train_fastgl.py --dataset_name  {} \
                --model {} --batch_size {} --log_path logs_fig15b/".format(data, model, bs)		
    exit_code = os.system(command)
print('*********Running experiments in fig 15b complete!*********\n')