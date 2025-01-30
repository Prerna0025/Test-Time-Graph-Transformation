import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parameter as Parameter
from torch_sparse import SparseTensor,matmul
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import pickle as pkl
import os

class GCN_gen(nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers):
        super(GCN_gen,self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels,hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(
                GCNConv(hidden_channels,hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels,out_channels))
        
        self.activation = F.relu
        
    def reset_parameter(self):
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self,x, edge_index, edge_weight=None):
        for i,conv in enumerate(self.convs[:-1]):
            x = conv(x,edge_index,edge_weight)
            x = self.activation(x)
        x = self.convs[-1](x,edge_index)
        return x 
    
def gen_amazon_dataset(name,model='gcn'):
    from torch_geometric.datasets import Amazon
    torch_dataset = Amazon(root=f'../data/Amazon',name=name)
    data = torch_dataset[0]
    edge_index = data.edge_index
    x = data.x
    label = data.y
    c = label.max().item()+1
    d=x.shape[1]

    data_dir = '../data/Amazon/{}/gen'.format(name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)                   
        
    Generator_x = GCN_gen(10,10,10,2)
    Generator_y = GCN_gen(in_channels=d,hidden_channels=10,out_channels=10,num_layers=2)
    Generator_noise = nn.Linear(10,10)
    for i in range(10):
        x_new = x
        y_new = Generator_y(x,edge_index)
        y_new = torch.argmax(y_new, dim=-1)
        label_new = F.one_hot(y_new,10).squeeze(1).float()
        context_ = torch.zeros(x.size(0),10)
        context_[:,i]=1
        x2 = Generator_x(label_new,edge_index)+Generator_noise(context_)
        x_new = torch.cat([x_new,x2],dim=1)
        
        with open(data_dir+'/{}-{}.pkl'.format(i,model),'wb') as f:
            pkl.dump((x_new,y_new),f,pkl.HIGHEST_PROTOCOL)
            
if __name__ == '__main__':
    gen_amazon_dataset(name='photo',model='gat')