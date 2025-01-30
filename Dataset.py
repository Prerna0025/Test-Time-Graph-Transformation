import numpy as np
import torch
import torch.nn.functional as F
from os import path
import pickle as pkl
from torch_geometric.datasets import Planetoid, Amazon

class NCDataset(object):
    def __init__(self,name):
        self.name = name
        self.graph = {}
        self.label = None
        
    def __getitem__(self,idx):
        assert idx ==0
        return self.graph, self.label
    
    def __len__(self):
        return 1
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,len(self))
    
def load_nc_dataset(data_dir,dataname,sub_dataname='',gen_model='gcn'):
    dataset = load_synthetic_dataset(data_dir,dataname,sub_dataname,gen_model)
    return dataset

def load_synthetic_dataset(data_dir,name,lang,gen_model='gcn'):
    dataset = NCDataset(lang)
    
    assert lang in range(0,10),'Invalid dataset'
    
    node,feat,y = pkl.load(open('{}/Amazon/Photo/gen/{}-{}.pkl'.format(data_dir,lang,gen_model),'rb'))
    torch_dataset = Amazon(root='{}/Amazon'.format(data_dir),name='Photo')
    data = torch_dataset[0]
    
    edge_index = data.edge_index
    label=y
    num_nodes = node_feat.size(0)
    
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat':None,
                     'num_nodes': num_nodes}
    
    dataset.label = label
    return dataset