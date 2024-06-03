#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


# model = 'gin'
# hidden = [64]

para = [
        ('ogbn-products'      ,  'gcn', [5,10]),   
        ('ogbn-products'      ,  'gcn', [5,10,15]), 
        ('ogbn-products'      ,  'gcn', [5,5,10,10]),
]

print('*********Running experiments in fig 15d begin*********\n')
for data, model, fanout in para:
    if(len(fanout)==2):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --fanout 5 10 --log_path logs_fig15d/".format(data, model)		
        exit_code = os.system(command)
    elif(len(fanout)==3):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --fanout 5 10 15 --log_path logs_fig15d/".format(data, model)		
        exit_code = os.system(command)
    elif(len(fanout)==4):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --fanout 5 5 10 10 --log_path logs_fig15d/".format(data, model)		
        exit_code = os.system(command)
print('*********Running experiments in fig 15d complete!*********\n')