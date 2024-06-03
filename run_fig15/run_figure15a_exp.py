#!/usr/bin/env python3
import os
os.environ["PYTHONWARNINGS"] = "ignore"


# model = 'gin'
# hidden = [64]

para = [
        ('ogbn-products'      ,  'gcn', [0]) ,   
        ('ogbn-products'      ,  'gcn', [0,1]) , 
        ('ogbn-products'      ,  'gcn', [0,1,2,3]) ,
        ('ogbn-products'      ,  'gcn', [0,1,2,3,4,5]) ,
        ('ogbn-products'      ,  'gcn', [0,1,2,3,4,5,6,7]) ,   
]

print('*********Running experiments in fig 15a begin*********\n')
for data, model, devices in para:
    if(len(devices)==1):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --devices 0 --log_path logs_fig15a/".format(data, model)		
        os.system(command)
    elif(len(devices)==2):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --devices 0 1 --log_path logs_fig15a/".format(data, model)		
        os.system(command)
    elif(len(devices)==4):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --devices 0 1 2 3 --log_path logs_fig15a/".format(data, model)		
        os.system(command)
    elif(len(devices)==6):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --devices 0 1 2 3 4 5 --log_path logs_fig15a/".format(data, model)		
        os.system(command)
    elif(len(devices)==8):
        command = "python ../train_fastgl.py --dataset_name  {} \
                    --model {} --devices 0 1 2 3 4 5 6 7 --log_path logs_fig15a/".format(data, model)		
        os.system(command)
print('*********Running experiments in fig 15a complete!*********\n')