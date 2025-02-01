import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
from torch_geometric.nn import GCNConv
import numpy as np
import math

class GCN(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,num_layers, dropout,save_mem = True, use_bn=True):
        super(GCN, self).__init__()
    
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=not save_mem,normalize=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels,hidden_channels,cached = not save_mem, normalize=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels,out_channels,cached=not save_mem, normalize=True))
        
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def forward(self,x,edge_index,edge_weight=None):
        for i,conv in enumerate(self.convs[-1]):
            x = conv(x,edge_index,edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training = self.training)  
        x = self.convs[-1](x,edge_index)
        return x      
            
    