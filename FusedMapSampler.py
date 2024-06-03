#*************************************************************************
#   > Filename    : FusedMapSampler.py
#   > Description : The sampler realized by our Fused-Map approach
#*************************************************************************

from dgl.dataloading.base import BlockSampler

import dgl
from dgl.dataloading.base import EID,NID
import time
import fastgl
from utilis import *

def event_sync():
    event = torch.cuda.Event(blocking=True)
    event.record()
    event.synchronize()


    
def print_edges(edge_0,edge_1,src_edges_out,dst_edges_out):
    print("edge_0:",edge_0)
    print("edge_1:",edge_1)
    print("src_edges_out:",src_edges_out)
    print("dst_edges_out:",dst_edges_out)

class FusedMapSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, mask=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                    'Mask and probability arguments are mutually exclusive. '
                    'Consider multiplying the probability with the mask '
                    'to achieve the same goal.')
        self.prob = prob or mask
        self.replace = replace
        
        self.nodes_num = 0
        
        # 统计构建子图和id转换的时间
        
        self.time_construct = 0.0
        self.time_idtrans = 0.0
        self.time_unique = 0.0
        self.time_create = 0.0
        self.time_index = 0.0
        
        self.real_block = []
        self.output_nodes = None
        self.seed_nodes = None
        
        self.edge_src = None
        self.edge_dst = None
        self.edge_ptr = None
        
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # seed_nodes = seed_nodes.sort()[0]
        self.output_nodes = seed_nodes
        raw_seed_nodes = seed_nodes.int().clone()
        seed_nodes_acc = seed_nodes.clone()
        blocks = []
        blocks_ = []
        self.real_block.clear()
        dst_nodes = seed_nodes.size(0)
        for block_index, fanout in enumerate(reversed(self.fanouts)):
            edge_src,edge_dst,edge_ptr = self.edge_src, self.edge_dst, self.edge_ptr
            
            # sample subgraph
            src_edges_out,dst_edges_out,edge_num = fastgl.sample_node(edge_src,edge_dst,edge_ptr,seed_nodes_acc.int(),dst_nodes,fanout)
            
            event_sync()
            
            # the number of the edges in a subgraph
            edge_nums = src_edges_out.size(0)
            
            
            
            if(block_index==0):
                dst_edges_unique = raw_seed_nodes
            else:
                dst_edges_unique = seed_nodes_acc
            
            if(block_index==0):
                output_nodes = raw_seed_nodes
            
            
            # max_src_num is the max number of the unique src node id 
            max_src_num = self.nodes_num if (fanout*dst_edges_unique.size(0)>self.nodes_num) else (fanout*dst_edges_unique.size(0))
            
            # fused-map process; 
            src_edges, dst_edges, seed_nodes_acc = fastgl.transfer_edge_id(src_edges_out,dst_edges_out,dst_edges_unique,\
                dst_nodes,dst_nodes + edge_nums,max_src_num,edge_nums)
            
            # get the output subgraph
            create_block = dgl.create_block((src_edges,dst_edges),num_src_nodes=seed_nodes_acc.size(0),num_dst_nodes=dst_edges_unique.size(0))
            
            event_sync()
            
            blocks.insert(0, create_block)
            dst_nodes = create_block.num_src_nodes()

        return seed_nodes_acc, output_nodes, blocks
    
class DGLNeighborSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, mask=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                    'Mask and probability arguments are mutually exclusive. '
                    'Consider multiplying the probability with the mask '
                    'to achieve the same goal.')
        self.prob = prob or mask
        self.replace = replace
        self.time_construct = 0.0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            t_construct_start = time.time()
            eid = frontier.edata[EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
            t_construct_end = time.time()
            
            self.time_construct += (t_construct_end - t_construct_start)

        return seed_nodes, output_nodes, blocks