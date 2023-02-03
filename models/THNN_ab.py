import torch
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
Rank = 20

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return (n*factorial(n-1))


class HyperGraph:
    def __init__(self, H):
        self.node = OrderedDict()
        self.edge = OrderedDict()
        self.max_order = int(torch.max(torch.Tensor(H).sum(dim=0)))

        max_node = 0
        min_node = 100

        print('Generating hypergraph from incidence matrix.')
        for i in range(H.shape[0]):
            nodename = i
            edge_index = np.argwhere(H[i] == 1).reshape(-1)
            self.node.update({nodename:edge_index})

        for i in range(H.shape[1]):
            edgename = i
            index = np.argwhere(H[:,i] == 1).reshape(-1)
            if len(index) > max_node:
                max_node = len(index)
            if len(index) < min_node:
                min_node = len(index)
            
            self.edge.update({edgename:index})
        print('Max_node in one edge',max_node)
        print('Min_node in one edge',min_node)


from collections import OrderedDict
def sort_key(old_dict, reverse=False):

    keys = sorted(old_dict.keys(), reverse=reverse)

    new_dict = OrderedDict()

    for key in keys:
        new_dict[key] = old_dict[key]
    return new_dict

class HyperGraph2:
    def __init__(self, hypergraph):
        self.node = OrderedDict()
        self.edge = OrderedDict()

        print('Generating hypergraph from hypergraph')
        num = 0
        for i in hypergraph.keys():
            edgename = num
            node_indexs = hypergraph[i]
            self.edge.update({edgename:np.array(list(node_indexs)).astype(int)})

            for j in node_indexs:
                if j in self.node.keys():
                    self.node[j].append(num)
                else:
                    self.node.update({j:[num]})
            num += 1
        for j in self.node.keys():
            self.node[j] = np.array(self.node[j])
        self.node  = sort_key(self.node)  

class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class, dropout = 0.7):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(n_hid, n_class),
        )
        

    def forward(self, x):
        x = self.fc1(x)
        return x

                 



class THNN_layer_notanh(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, rank=50, dropout = 0.3):
        super(THNN_layer_notanh, self).__init__()
        self.featdim = featdim
        self.rank = rank


        self.p_network = nn.Sequential(
            nn.Linear(featdim + 1, rank),
            nn.Dropout(dropout)
        )
        self.q_network = nn.Sequential(
            nn.Linear(rank, outputdim),
            nn.Dropout(dropout)
        )
        self.p2_network = nn.Sequential(
            nn.Linear(featdim + 1, hiddendim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hiddendim, outputdim),
        )
        
        for m in self.p_network:
             if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(1/(featdim + 1)))
                nn.init.constant_(m.bias, 0)        

    def forward(self, HyperGraph, embedding):
        

        one = torch.ones(embedding.shape[0]).to(device)
        embedding = torch.cat((embedding, one.reshape(-1, 1)), 1) # Adding ones
        embedding_new = self.p_network(embedding)
        embedding_new2 = self.p2_network(embedding)
        fea_dim_tensor = 0

        for node in HyperGraph.node.keys():  # all nodes
            start1 = time.time()
            node_emb = 0
            node_emb_stack = 0
            node_emb2_stack = 0
            for edge in HyperGraph.node[node]:  # for edges in node
                edge_emb = torch.ones(self.rank).to(device)
                edge_emb2 = 0
                node_num = len(HyperGraph.edge[edge])  # nodenum in one edge
                
                for edge_node in HyperGraph.edge[edge]:
                    if int(node) != edge_node: # no self-loop
                        degree = len(HyperGraph.node[edge_node])
                        num = degree ** (1 / node_num)
                        edge_emb = torch.einsum('r,r->r', edge_emb, num * embedding_new[int(edge_node)].reshape(-1))
                    edge_emb2 += embedding_new2[int(edge_node)].reshape(-1)

                # node_emb += self.q_network(F.tanh((1/factorial(node_num-1)) * edge_emb))
                degree = len(HyperGraph.node[node])
                num = degree ** (1 / node_num)
                node_emb =(1 / math.factorial(node_num - 1)) * num * edge_emb
                node_emb = node_emb.reshape(1, -1)
                edge_emb2 = edge_emb2.reshape(1, -1)
        


                if type(node_emb_stack) != torch.Tensor:
                    node_emb_stack = node_emb
                else:
                    node_emb_stack = torch.cat((node_emb_stack, node_emb), 0)

                if type(node_emb2_stack) != torch.Tensor:
                    node_emb2_stack = edge_emb2
                else:
                    node_emb2_stack = torch.cat((node_emb2_stack, edge_emb2), 0)
                
            node_emb = (self.q_network(node_emb_stack) + F.relu(node_emb2_stack)).mean(dim=0)
            node_emb = node_emb.reshape(1, -1)

            if type(node_emb) != torch.Tensor:
                node_emb = embedding_new[int(node)]

            if type(fea_dim_tensor) != torch.Tensor:
                fea_dim_tensor = node_emb
            else:
                fea_dim_tensor = torch.cat((fea_dim_tensor, node_emb), 0)
            start2 = time.time()
        return F.relu(fea_dim_tensor)



class THNN_layer_no_addingones(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, rank=50, dropout = 0.3):
        super(THNN_layer_no_addingones, self).__init__()
        self.featdim = featdim
        self.rank = rank


        self.p_network = nn.Sequential(
            nn.Linear(featdim , rank),
            nn.Dropout(dropout)
        )
        self.q_network = nn.Sequential(
            nn.Linear(rank, outputdim),
            nn.Dropout(dropout)
        )
        self.p2_network = nn.Sequential(
            nn.Linear(featdim , hiddendim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hiddendim, outputdim),
        )
        
        for m in self.p_network:
             if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(1/(featdim + 1)))
                nn.init.constant_(m.bias, 0)        

    def forward(self, HyperGraph, embedding):
        
        embedding_new = self.p_network(embedding)
        embedding_new2 = self.p2_network(embedding)
        fea_dim_tensor = 0

        for node in HyperGraph.node.keys():  # all nodes
            start1 = time.time()
            node_emb = 0
            node_emb_stack = 0
            node_emb2_stack = 0
            for edge in HyperGraph.node[node]:  # for edges in node
                edge_emb = torch.ones(self.rank).to(device)
                edge_emb2 = 0
                node_num = len(HyperGraph.edge[edge])  # nodenum in one edge
                
                for edge_node in HyperGraph.edge[edge]:
                    if int(node) != edge_node: # no self-loop
                        degree = len(HyperGraph.node[edge_node])
                        num = degree ** (1 / node_num)
                        edge_emb = torch.einsum('r,r->r', edge_emb, num * embedding_new[int(edge_node)].reshape(-1))
                    edge_emb2 += embedding_new2[int(edge_node)].reshape(-1)

                degree = len(HyperGraph.node[node])
                num = degree ** (1 / node_num)
                node_emb =F.tanh((1 / math.factorial(node_num - 1)) * num * edge_emb) 
                node_emb = node_emb.reshape(1, -1)
                edge_emb2 = edge_emb2.reshape(1, -1)
        


                if type(node_emb_stack) != torch.Tensor:
                    node_emb_stack = node_emb
                else:
                    node_emb_stack = torch.cat((node_emb_stack, node_emb), 0)

                if type(node_emb2_stack) != torch.Tensor:
                    node_emb2_stack = edge_emb2
                else:
                    node_emb2_stack = torch.cat((node_emb2_stack, edge_emb2), 0)
                
            node_emb = (self.q_network(node_emb_stack) + F.relu(node_emb2_stack)).mean(dim=0)
            node_emb = node_emb.reshape(1, -1)

            if type(node_emb) != torch.Tensor:
                node_emb = embedding_new[int(node)]

            if type(fea_dim_tensor) != torch.Tensor:
                fea_dim_tensor = node_emb
            else:
                fea_dim_tensor = torch.cat((fea_dim_tensor, node_emb), 0)
            start2 = time.time()
        return F.relu(fea_dim_tensor)

class THNN_layer_no_normalizations(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, rank=50, dropout = 0.3):
        super(THNN_layer_no_normalizations, self).__init__()
        self.featdim = featdim
        self.rank = rank


        self.p_network = nn.Sequential(
            nn.Linear(featdim + 1, rank),
            nn.Dropout(dropout)
        )
        self.q_network = nn.Sequential(
            nn.Linear(rank, outputdim),
            nn.Dropout(dropout)
        )
        self.p2_network = nn.Sequential(
            nn.Linear(featdim + 1, hiddendim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hiddendim, outputdim),
        )
        
        for m in self.p_network:
             if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(1/(featdim + 1)))
                nn.init.constant_(m.bias, 0)        

    def forward(self, HyperGraph, embedding):
        

        one = torch.ones(embedding.shape[0]).to(device)
        embedding = torch.cat((embedding, one.reshape(-1, 1)), 1)
        embedding_new = self.p_network(embedding)
        embedding_new2 = self.p2_network(embedding)
        fea_dim_tensor = 0

        for node in HyperGraph.node.keys():  # all nodes
            start1 = time.time()
            node_emb = 0
            node_emb_stack = 0
            node_emb2_stack = 0
            for edge in HyperGraph.node[node]:  # for edges in node
                edge_emb = torch.ones(self.rank).to(device)
                edge_emb2 = 0
                node_num = len(HyperGraph.edge[edge])  # nodenum in one edge
                
                for edge_node in HyperGraph.edge[edge]:
                    if int(node) != edge_node: # no self-loop
                        degree = len(HyperGraph.node[edge_node])
                        num = 1
                        edge_emb = torch.einsum('r,r->r', edge_emb, num * embedding_new[int(edge_node)].reshape(-1))
                    edge_emb2 += embedding_new2[int(edge_node)].reshape(-1)

                # node_emb += self.q_network(F.tanh((1/factorial(node_num-1)) * edge_emb))
                degree = len(HyperGraph.node[node])
                num = 1
                node_emb =F.tanh((1 / math.factorial(node_num - 1)) * num * edge_emb) 
                node_emb = node_emb.reshape(1, -1)
                edge_emb2 = edge_emb2.reshape(1, -1)
        


                if type(node_emb_stack) != torch.Tensor:
                    node_emb_stack = node_emb
                else:
                    node_emb_stack = torch.cat((node_emb_stack, node_emb), 0)

                if type(node_emb2_stack) != torch.Tensor:
                    node_emb2_stack = edge_emb2
                else:
                    node_emb2_stack = torch.cat((node_emb2_stack, edge_emb2), 0)
                
            node_emb = (self.q_network(node_emb_stack) + F.relu(node_emb2_stack)).mean(dim=0)
            node_emb = node_emb.reshape(1, -1)

            if type(node_emb) != torch.Tensor:
                node_emb = embedding_new[int(node)]

            if type(fea_dim_tensor) != torch.Tensor:
                fea_dim_tensor = node_emb
            else:
                fea_dim_tensor = torch.cat((fea_dim_tensor, node_emb), 0)
            start2 = time.time()
        return F.relu(fea_dim_tensor)






class THNN_ab(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, n_class, rank = Rank, dropout = 0.3, mode = 0):
        super(THNN_ab, self).__init__()
        self.dropout = dropout
        if mode == 0:
            self.model1 = THNN_layer_notanh( featdim, hiddendim, hiddendim, rank, dropout)
            self.model2 = THNN_layer_notanh( outputdim, hiddendim, hiddendim, rank, dropout)
        elif mode == 1:
            self.model1 = THNN_layer_no_addingones( featdim, hiddendim, hiddendim, rank, dropout)
            self.model2 = THNN_layer_no_addingones( outputdim, hiddendim, hiddendim, rank, dropout)
        elif mode == 2:
            self.model1 = THNN_layer_no_normalizations( featdim, hiddendim, hiddendim, rank, dropout)
            self.model2 = THNN_layer_no_normalizations( outputdim, hiddendim, hiddendim, rank, dropout)
        self.classifier = HGNN_classifier( hiddendim, n_class)


    def forward(self,HyperGraph, embedding, N=2):
        
        embedding = (embedding - embedding.mean())/(embedding.std())
        start1 = time.time()

        embedding_ = self.model1.forward(HyperGraph,  embedding)
        if N>1:
            embedding_ = F.dropout(F.relu(embedding_), self.dropout)
            embedding_ = self.model2.forward(HyperGraph, embedding_)
        if N>2:
            embedding_ = F.dropout(F.relu(embedding_), self.dropout)
        output = self.classifier(embedding_)
        return output
    
