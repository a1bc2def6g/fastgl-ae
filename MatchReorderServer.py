#*************************************************************************
#   > Filename    : MatchReorderServer.py
#   > Description : Construct the gnn layer and model with the Memory-Aware(MA) computation  method
#*************************************************************************

import os
import sys

import torch
        
        
class MatchServer:
    def __init__(self,node_num,gpuid,high_nid,cache_ratio):
        
        self.gpuid = gpuid
        self.node_num = node_num
        # the portion of all nodes that cached on GPU
        self.cache_ratio = cache_ratio
        
        # masks for manage the feature locations: default in CPU
        self.gpu_flag = torch.zeros(self.node_num,device=self.gpuid).bool()
        self.gpu_flag.requires_grad_(False)
        
        self.dynamic_gpu_flag = None
        
        # used to record the global id of the current batch
        self.gpu_batch_flag = torch.zeros(self.node_num,device=self.gpuid).bool()
        self.gpu_batch_flag.requires_grad_(False)
        
        with torch.cuda.device(self.gpuid):
            self.localid2cacheid = torch.cuda.IntTensor(node_num).fill_(0)
            self.localid2cacheid.requires_grad_(False)
            
            self.localid2cacheid_batch = torch.cuda.IntTensor(node_num).fill_(0)
            self.localid2cacheid_batch.requires_grad_(False)
            
        self.high_nid = high_nid
        
        self.use_cache = False
        
    def init_cache(self, input_nodes, input):
        # get available GPU memory
        peak_allocated_mem = torch.cuda.max_memory_allocated(device=self.gpuid)
        peak_cached_mem = torch.cuda.max_memory_reserved(device=self.gpuid)
        total_mem = torch.cuda.get_device_properties(self.gpuid).total_memory
        available = total_mem - peak_allocated_mem - peak_cached_mem \
                    - 2 * 1024 * 1024 * 1024 # in bytes
        # get capability
        # if cache_ratio>1, cache the nodes according to the gpu cacpacity
        # else, cache the nodes according to the cache_ratio
        if(self.cache_ratio>1):
            self.capability = int(available / (input.size(1) * 4)) # assume float32 = 4 bytes
        else:
            self.capability = int(self.node_num*self.cache_ratio)
        
        # update gpu cache
        if(self.capability>=self.node_num):
            self.use_cache = True
            self.memory = input.cuda(self.gpuid)
            rows = input.size(0)
            self.localid2cacheid = torch.arange(rows,device=self.gpuid).int()
            self.gpu_flag[:] = True
        elif((self.capability<self.node_num)&(self.capability>0)):
            self.use_cache = True
            fetch_nid = self.high_nid[:self.capability]
            self.memory = torch.index_select(
                        input, 0, fetch_nid.long()).to(self.gpuid)
            rows = fetch_nid.size(0)
            self.localid2cacheid[fetch_nid] = torch.arange(rows,device=self.gpuid).int()
            self.gpu_flag[fetch_nid] = True
        else:
            self.use_cache = False
            
        self.gpu_batch_flag[input_nodes] = True
        
        self.localid2cacheid_batch[input_nodes] = torch.arange(input_nodes.size(0),device=self.gpuid).int()
        
    
    def fetch_cpu_data_id(self,input_nodes):
        if(self.use_cache):
            gpu_mask = self.gpu_flag[input_nodes]
            nids_in_gpu = input_nodes[gpu_mask]
            cacheid = self.localid2cacheid[nids_in_gpu]
        else:
            gpu_mask = None
            cacheid = None
        
        # 
        gpu_batch_mask = self.gpu_batch_flag[input_nodes]
        nids_in_last_batch = input_nodes[gpu_batch_mask]
        
        if(self.use_cache):
            cpu_mask = ~(gpu_mask | gpu_batch_mask)
        else:
            cpu_mask = ~gpu_batch_mask
        nids_in_cpu = input_nodes[cpu_mask]
        
        cacheid_batch = self.localid2cacheid_batch[nids_in_last_batch]
        
        return cacheid, cacheid_batch, nids_in_cpu, gpu_mask, gpu_batch_mask, cpu_mask
    
    def fresh_cache(self,input_nodes,input_feat):
            
        self.gpu_flag.fill_(False)
        self.gpu_flag[input_nodes.long()] = True
            
            
        self.localid2cacheid_batch.fill_(0)
        rows = input_feat.size(0)
        self.localid2cacheid_batch[input_nodes.long()] = torch.arange(rows,device=self.gpuid).int()
            
            
    
        
    
    
    
    
