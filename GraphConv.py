#*************************************************************************
#   > Filename    : GraphConv.py
#   > Description : Construct the gnn layer and model with the Memory-Aware(MA) computation  method
#*************************************************************************

from typing import Any
import torch
from torch import nn
from torch.nn import init
import dgl.function as fn
from dgl.utils import expand_as_pair
import fastgl
from utilis import *
from dgl.nn.functional import edge_softmax
import math
import time

    
class MA_Function(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, edge_ptr, src_edges, src_norm_degs, dst_norm_degs, dst_nodes, input_feat, weight, neighbor_num):
        
        
        ctx.save_for_backward(edge_ptr,src_edges,src_norm_degs,dst_norm_degs,weight,input_feat)
        
        X_prime = fastgl.forward_gcn(edge_ptr,src_edges,src_norm_degs,dst_norm_degs,input_feat,weight,neighbor_num,
                                     dst_nodes)[0]
    

        return X_prime
    
    @staticmethod
    def backward(ctx, d_input):
        
        edge_ptr, src_edges, src_norm_degs, dst_norm_degs, weight, input_feat = ctx.saved_tensors
        
        d_x,d_w = fastgl.backward_gcn(edge_ptr,src_edges,dst_norm_degs,src_norm_degs,d_input, weight,
                                             input_feat,)
        
        return None, None, None, None, None, d_x, d_w, None

    
class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, weight=False):
        super(GCNConv, self).__init__()
        if(weight):
            self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        else:
            self.weights = self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if(self.weights!=None):
            stdv = 1. / math.sqrt(self.weights.size(1))
            self.weights.data.uniform_(-stdv, stdv)

    def forward(self, edge_ptr, src_edges, src_norm_degs, dst_norm_degs, dst_nodes, input_feat, neighbor_num):
        if(self.weights is not None):
            weight = self.weights
        
        return MA_Function.apply(edge_ptr, src_edges, src_norm_degs, dst_norm_degs, dst_nodes, input_feat, weight, neighbor_num)
    
class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 weight,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GCNConv(in_feats, n_hidden,weight=weight))
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GCNConv(n_hidden, n_hidden,weight=weight))
        # output layer
        self.layers.append(
            GCNConv(n_hidden, n_classes,weight=weight))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features, neighbor_list=[5,10,15]):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            graph = blocks[i]
            dst_nodes = graph.num_dst_nodes()
            raw_dst_norm_degs = fastgl.cal_deg(graph.edges()[1].int(),dst_nodes)
            edge_ptr = fastgl.exclusive_sum(raw_dst_norm_degs)
            
            src_norm_degs = fastgl.cal_deg(graph.edges()[0].int(),graph.num_src_nodes()).clamp(min=1)
            dst_norm_degs = raw_dst_norm_degs.clamp(min=1)
            h = layer(edge_ptr, graph.edges()[0].int(), src_norm_degs, dst_norm_degs, dst_nodes, h, neighbor_list[i])
        return h
    
    
class MA_Function_GIN(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, edge_ptr, src_edges, dst_nodes, input_feat, weight, neighbor_num):
        
        # edge_ptr, src_edges, src_norm_degs, dst_norm_degs, dst_nodes = get_param(graph)
        
        
        X_prime, tmp = fastgl.forwaed_gin(edge_ptr,src_edges,input_feat,weight,neighbor_num,
                                     dst_nodes)
        # X_prime = input_feat[:dst_nodes,:weight.size(1)]
        
        ctx.src_nodes = input_feat.size(0)
        ctx.neighbor_num = neighbor_num
    

        ctx.save_for_backward(edge_ptr,src_edges,weight,tmp)
        
        return X_prime
    
    @staticmethod
    def backward(ctx, d_input):
        
        edge_ptr, src_edges, weight, input_feat = ctx.saved_tensors
        
        src_nodes = ctx.src_nodes
        # if(ctx.neighbor_num==5):
        #     print("stop",a)
        
        # print("stop",a)
        
        d_x,d_w = fastgl.backward_gin(edge_ptr,src_edges,d_input, weight,
                                             input_feat,src_nodes)
        
        return None, None, None, d_x, d_w, None

    
class GINConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, weight=False):
        super(GINConv, self).__init__()
        if(weight):
            self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        else:
            self.weights = self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        if(self.weights!=None):
            stdv = 1. / math.sqrt(self.weights.size(1))
            self.weights.data.uniform_(-stdv, stdv)

    def forward(self, edge_ptr, src_edges, dst_nodes, input_feat, neighbor_num):
        if(self.weights is not None):
            weight = self.weights
        
        return MA_Function_GIN.apply(edge_ptr, src_edges, dst_nodes, input_feat, weight, neighbor_num)
    

class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 weight,
                 dropout):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GINConv(in_feats, n_hidden,weight=weight))
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GINConv(n_hidden, n_hidden,weight=weight))
        # output layer
        self.layers.append(
            GINConv(n_hidden, n_classes,weight=weight))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features, neighbor_list=[5,10,15]):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            graph = blocks[i]
            dst_nodes = graph.num_dst_nodes()
            dst_edges = graph.edges()[1].int()
            raw_dst_norm_degs = fastgl.cal_deg(dst_edges,dst_nodes)
            edge_ptr = fastgl.exclusive_sum(raw_dst_norm_degs)
            
            src_edges = graph.edges()[0].int()
            h = layer(edge_ptr, src_edges, dst_nodes, h, neighbor_list[i])
        return h 
    
    
class MA_Function_GAT(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, edge_ptr, edge_data, src_edges, dst_edges, dst_nodes, input_feat, num_heads, neighbor_num):
        
        
        
        X_prime = fastgl.forward_gat(edge_ptr,edge_data,src_edges,input_feat,neighbor_num,
                                     dst_nodes,num_heads)[0]
        
        ctx.src_nodes = input_feat.size(0)
        ctx.num_heads = num_heads
        ctx.neighbor_num = neighbor_num
    

        ctx.save_for_backward(edge_ptr,edge_data,src_edges, dst_edges, input_feat)
        
        return X_prime
    
    @staticmethod
    def backward(ctx, d_input):
        
        edge_ptr, edge_data, src_edges, dst_edges, input_feat = ctx.saved_tensors
        
        src_nodes = ctx.src_nodes
        num_heads = ctx.num_heads
        
        d_x = fastgl.backward_gat(edge_ptr,edge_data,src_edges,d_input,
                                             src_nodes,num_heads)[0]
        
        return None, None, None, None, None, d_x, None, None

    
class GATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, feat_drop=0, attn_drop=0, negative_slope=0.2):
        super(GATConv, self).__init__()
        self.out_feats = output_dim
        self.num_heads = num_heads
        self.fc = nn.Linear(
        input_dim, output_dim * num_heads, bias=False)
        
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_dim)))
        # self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 2*output_dim)))
        # self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_dim)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, graph, input_feat, neighbor_num):
        dst_nodes = graph.num_dst_nodes()
        feat_src = feat_dst = self.fc(input_feat).view(
            -1, self.num_heads, self.out_feats)
        feat_dst = feat_src[:dst_nodes]
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        edge_data = self.attn_drop(edge_softmax(graph, e)).squeeze(-1)
        
        feat_src = feat_src.view(-1, self.out_feats*self.num_heads)
        
        
        
        
        raw_dst_norm_degs = fastgl.cal_deg(graph.edges()[1].int(),dst_nodes)
        edge_ptr = fastgl.exclusive_sum(raw_dst_norm_degs)
        
        return MA_Function_GAT.apply(edge_ptr, edge_data, graph.edges()[0].int(), graph.edges()[1].int(), dst_nodes, feat_src, self.num_heads, neighbor_num)
    
    
class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 heads=8):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_feats,n_hidden,num_heads=heads,))
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(GATConv(n_hidden*heads,n_hidden,num_heads=heads,feat_drop=dropout))
        # output layer
        self.layers.append(GATConv(n_hidden*heads,n_classes,num_heads=1,feat_drop=dropout))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features, neighbor_list=[5,10,15]):
        h = features
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(blocks[i], h, neighbor_list[i])
            # h = h.view(h.size(0),-1)
        return h
