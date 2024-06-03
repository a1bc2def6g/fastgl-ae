#*************************************************************************
#   > Filename    : dgl_model.py
#   > Description : dgl baseline model
#*************************************************************************
import torch.nn as nn
from dgl.nn.pytorch import GraphConv,GINConv,GATConv
# import fastgraph
from common_config import *
from utilis import *




class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        # output layer
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h

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
        self.layers.append(GATConv(in_feats,n_hidden,num_heads=heads,allow_zero_in_degree=True,bias=False))
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(GATConv(n_hidden*heads,n_hidden,num_heads=heads,feat_drop=0,allow_zero_in_degree=True,bias=False))
        # output layer
        self.layers.append(GATConv(n_hidden*heads,n_classes,num_heads=1,feat_drop=0,allow_zero_in_degree=True,bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            # if i != 0:
            #     h = self.dropout(h)
            h = layer(blocks[i], h)
            h = h.view(h.size(0),-1)
        return h
    
class GIN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Linear(in_feats, n_hidden, bias=False),
                    # nn.ReLU(),
                ), 
                aggregator_type='sum',
                learn_eps=False)
        )
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GINConv(nn.Sequential(
                    nn.Linear(n_hidden, n_hidden, bias=False),
                    # nn.ReLU(),
                    ), 
                    aggregator_type='sum',
                )
            )
        # output layer
        self.layers.append(
            GINConv(nn.Sequential(
                        nn.Linear(n_hidden, n_classes, bias=False),
                        # nn.ReLU(),
                        ),
                    aggregator_type='sum',
                    ),
            
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h