#*************************************************************************
#   > Filename    : utils.py
#   > Description : some utils function
#*************************************************************************
import torch
import os
from dgl.data.utils import load_graphs
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.distributed import partition_graph,load_partition
    
class match_reorder():
    def __init__(self, num_nodes, steps, device):
        """_summary_

        Args:
            num_nodes (_type_): the number of the nodes in the entire graph
            steps (_type_): the number of the n
            device (_type_): on which gpu
        """
        self.steps = steps
        self.device = device
        self.node_num = num_nodes
        self.gpu_flag = []
        self.sample_num = torch.zeros(self.node_num).cuda(device)
        self.match_scator = torch.zeros((self.steps,self.steps),device=self.device)
        for i in range(self.steps):
            self.gpu_flag.append(torch.zeros(num_nodes).bool().cuda(self.device))
        
        
    def reorder(self,dataloader):
        """realize the Greedy Reorder Strategy
        """
        batch_list = []
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            if(step>=self.steps):
                break
            self.gpu_flag[step][input_nodes.to(torch.long)] = True
            batch_list.append((input_nodes, output_nodes, blocks))
        
        return_batch_list = []
        return_batch_list.append(batch_list[0])
        # cal the match degrees between different subgraphs
        for i in range(self.steps):
            for j in range(i+1, self.steps):
                self.match_scator[i][j] = (self.gpu_flag[i] & self.gpu_flag[j]).sum()/(self.gpu_flag[j].sum())
                self.match_scator[j][i] = self.match_scator[i][j]
            self.gpu_flag[i].fill_(False)
        last_batch = 0
        
        # reorder by the Greedy Reorder Strategy
        
        for i in range(1, self.steps):
            index = torch.argmax(self.match_scator[last_batch,:])
            return_batch_list.append(batch_list[index])
            
            # avoid adding the same subgraph into the list
            self.match_scator[last_batch,:] = 0
            self.match_scator[:,last_batch] = 0
            last_batch = index
            
        return return_batch_list
    
def pre_sample(dataloader,node_num):
    batch_list = []
    for step, (input_nodes, output_nodes, blocks) in enumerate(iter(dataloader)):
        batch_list.append(input_nodes)
    sample_num = torch.zeros(node_num).cuda(batch_list[0].device)
    for i in range(len(batch_list)):
        sample_num[batch_list[i].to(torch.long)] += 1
    _,indices = torch.sort(sample_num,descending=True)
    del batch_list
    return indices.cpu()


def presample_all_epoch(dataloader, node_num, epochs):
    sample_num = torch.zeros(node_num).cuda(dataloader.device)
    all_batch_list = [[] for i in range(epochs)]
    for i in range(epochs):
        for step, (input_nodes, output_nodes, blocks) in enumerate(iter(dataloader)):
            all_batch_list[i].append((input_nodes, output_nodes, blocks))
            sample_num[input_nodes] += 1
    _,indices = torch.sort(sample_num,descending=True)
    return all_batch_list, indices.cpu()

def sort_degree(g):
    degree = g.in_degrees()
    _,indices = torch.sort(degree,descending=True)
    return indices.cpu()


def get_param(graph):
    dst_nodes = graph.num_dst_nodes()
    ind = torch.stack((graph.edges()[1],graph.edges()[0]))
    val = ind.new_ones((graph.num_edges(),))
    size = [graph.num_dst_nodes(),graph.num_src_nodes()]
    edge_ptr = torch.sparse_coo_tensor(indices=ind, values=val, size=size).to_sparse_csr().crow_indices().int()
    src_edges = graph.edges()[0].int()
    src_norm_degs = graph.out_degrees().clamp(min=1).int()
    
    dst_norm_degs = graph.in_degrees().clamp(min=1).int()
    
    return edge_ptr, src_edges, src_norm_degs, dst_norm_degs, dst_nodes




def cal_only_in_src(src_edges,dst_edges_unique, raw_scr_edges):
    src_num = 0
    src_unique_nodes = []
    for i in range(src_edges.size(0)):
        if src_edges[i] not in dst_edges_unique:
            src_num += 1
            src_unique_nodes.append(src_edges[i])
        # else:
            # thread_id = torch.where(raw_scr_edges==src_edges[i])[0]
            # print("dst nodes in src:",src_edges[i].cpu().item(),thread_id.cpu())
    return src_num,src_unique_nodes

def print_blocks(blocks):
    for i,block in enumerate(blocks):
        print("block[{}]:{}".format(i,block))
        
        
def cmp_out_feat(out_dgl,out,sampler,blocks,index):
    acc_dst_nodes = blocks[-1].dstdata['_ID']
    dgl_dst_nodes = sampler.real_block[-1].srcdata['_ID']
    
    raw_id = dgl_dst_nodes[index]
    
    acc_id = torch.where(acc_dst_nodes==raw_id)[0]
    
    if(acc_id.size(0)!=0):
        print("dgl feat:",out_dgl[index])
        print("acc feat:",out[acc_id],"acc id:", acc_id)
    else:
        print("no raw id in add dst nodes")
    
