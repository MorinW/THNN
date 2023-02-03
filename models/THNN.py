import torch
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import time
Rank = 128

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
            nn.Dropout(dropout),
        )
        

    def forward(self, x):
        x = self.fc1(x)
        return x

                
class THNN_layer(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, rank=50, dropout = 0.3):
        super(THNN_layer, self).__init__()
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
        self.allign = nn.Sequential(
            nn.Linear(featdim + 1, outputdim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        
        for m in self.p_network:
             if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(1/(featdim + 1)))
                nn.init.constant_(m.bias, 0)        

    def forward(self, HyperGraph, embedding):
        

        one = torch.ones(embedding.shape[0]).to(device)
        embedding = torch.cat((embedding, one.reshape(-1, 1)), 1)
        resdiual = self.allign(embedding)
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
        return F.relu(fea_dim_tensor) + resdiual





class THNN(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, n_class, rank = Rank, dropout = 0.3):
        super(THNN, self).__init__()
        self.dropout = dropout
        self.model1 = THNN_layer( featdim, hiddendim, hiddendim, rank, dropout)
        self.model2 = THNN_layer( outputdim, hiddendim, hiddendim, rank, dropout)
        self.classifier = HGNN_classifier( hiddendim, n_class)


    def forward(self,HyperGraph, embedding):
        
        embedding = (embedding - embedding.mean())/(embedding.std())
        embedding_ = self.model1.forward(HyperGraph,  embedding)
        embedding_ = F.dropout(F.relu(embedding_), self.dropout)
        embedding_ = self.model2.forward(HyperGraph, embedding_)
        output = self.classifier(embedding_)
        return output
    
    
    
class THNN_order_layer(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, rank=50, dropout = 0.3, order = 3):
        super(THNN_order_layer, self).__init__()
        self.featdim = featdim
        self.rank = rank
        self.order = order


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
        self.allign = nn.Sequential(
            nn.Linear(featdim + 1, outputdim),
            nn.Dropout(dropout),
            nn.ReLU(),
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
        resdiual = self.allign(embedding)
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
                if node_num != self.order: # If node order is not the right, remove it.
                    continue
                
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
            if type(node_emb_stack) != torch.Tensor and self.order == 2:
                node_emb = embedding_new2[int(node)]
            elif type(node_emb_stack) != torch.Tensor and self.order != 2:
                node_emb = torch.zeros_like(embedding_new2[int(node)])
            else:
                node_emb = (self.q_network(node_emb_stack) + F.relu(node_emb2_stack)).mean(dim=0)
            
            node_emb = node_emb.reshape(1, -1)

            if type(node_emb) != torch.Tensor:
                node_emb = embedding_new[int(node)]

            if type(fea_dim_tensor) != torch.Tensor:
                fea_dim_tensor = node_emb
            else:
                fea_dim_tensor = torch.cat((fea_dim_tensor, node_emb), 0)

        return F.relu(fea_dim_tensor) + resdiual 


class THNN_2_3_4(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, n_class, rank = Rank, dropout = 0.3):
        super(THNN_2_3_4, self).__init__()
        
        self.order2 = THNN_order_layer(featdim, hiddendim, outputdim, rank, dropout = 0.3, order = 2)
        self.order3 = THNN_order_layer(featdim, hiddendim, outputdim, rank, dropout = 0.3, order = 3)
        self.order4 = THNN_order_layer(featdim, hiddendim, outputdim, rank, dropout = 0.3, order = 4)
        
        
        self.order22 = THNN_order_layer(outputdim * 3, hiddendim, outputdim, rank, dropout = 0.3, order = 2)
        self.order32 = THNN_order_layer(outputdim * 3, hiddendim, outputdim, rank, dropout = 0.3, order = 3)
        self.order42 = THNN_order_layer(outputdim * 3, hiddendim, outputdim, rank, dropout = 0.3, order = 3)
        self.classifier = HGNN_classifier( outputdim * 3, n_class)
    
    def forward(self,HyperGraph, embedding):
        embedding = (embedding - embedding.mean())/(embedding.std())
        e2 = self.order2.forward(HyperGraph, embedding)
        e3 = self.order3.forward(HyperGraph, embedding)
        e4 = self.order4.forward(HyperGraph, embedding)
        embedding = torch.cat((e2,e3,e4), 1)
        e2 = self.order22.forward(HyperGraph, embedding)
        e3 = self.order32.forward(HyperGraph, embedding)
        e4 = self.order42.forward(HyperGraph, embedding)
        embedding = torch.cat((e2,e3,e4), 1)        
        output = self.classifier.forward(embedding)
        
        return output
    

class THNN_global_layer(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, rank=50, dropout = 0.3, global_degree = 1):
        super(THNN_global_layer, self).__init__()
        self.featdim = featdim
        self.rank = rank
        self.maxorder = 4
        self.global_degree = global_degree


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
        self.allign = nn.Sequential(
            nn.Linear(featdim + 1, outputdim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        for m in self.p_network:
             if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(1/(featdim + 1)))
                nn.init.constant_(m.bias, 0)        

    def forward(self, HyperGraph, embedding, global_emb):
        
        
        one = torch.ones(embedding.shape[0]).to(device)
        embedding = torch.cat((embedding, one.reshape(-1, 1)), 1)
        resdiual = self.allign(embedding)
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
                if node_num == 1:
                    for edge_node in HyperGraph.edge[edge]:
                        if int(node) != edge_node: # no self-loop
                            degree = len(HyperGraph.node[edge_node])
                            num = degree ** (1 / node_num)
                            edge_emb = torch.einsum('r,r->r', edge_emb, num * embedding_new[int(edge_node)].reshape(-1))
                            # opt einsum
                        edge_emb =torch.einsum('r,r->r', (self.global_degree ** (1 / 4)) * edge_emb, global_emb)
                        edge_emb =torch.einsum('r,r->r', (self.global_degree ** (1 / 4)) * edge_emb, global_emb)
                        edge_emb =torch.einsum('r,r->r', (self.global_degree ** (1 / 4)) * edge_emb, global_emb)
                        edge_emb2 += embedding_new2[int(edge_node)].reshape(-1)
                elif node_num ==2:
                    for edge_node in HyperGraph.edge[edge]:
                        if int(node) != edge_node: # no self-loop
                            degree = len(HyperGraph.node[edge_node])
                            num = degree ** (1 / 3)
                            edge_emb = torch.einsum('r,r->r', edge_emb, num * embedding_new[int(edge_node)].reshape(-1))
                            # opt einsum
                        edge_emb2 += embedding_new2[int(edge_node)].reshape(-1)
                    edge_emb =torch.einsum('r,r->r', (self.global_degree ** (1 / 4)) * edge_emb, global_emb)
                    edge_emb =torch.einsum('r,r->r', (self.global_degree ** (1 / 4)) * edge_emb, global_emb)
                elif node_num ==3:
                    for edge_node in HyperGraph.edge[edge]:
                        if int(node) != edge_node: # no self-loop
                            degree = len(HyperGraph.node[edge_node])
                            num = degree ** (1 / 3)
                            edge_emb = torch.einsum('r,r->r', edge_emb, num * embedding_new[int(edge_node)].reshape(-1))
                            # opt einsum
                        edge_emb2 += embedding_new2[int(edge_node)].reshape(-1)
                    edge_emb =torch.einsum('r,r->r', (self.global_degree ** (1 / 4)) * edge_emb, global_emb)
                elif node_num ==4:
                    for edge_node in HyperGraph.edge[edge]:
                        if int(node) != edge_node: # no self-loop
                            degree = len(HyperGraph.node[edge_node])
                            num = degree ** (1 / node_num)
                            edge_emb = torch.einsum('r,r->r', edge_emb, num * embedding_new[int(edge_node)].reshape(-1))
                            # opt einsum
                        edge_emb2 += embedding_new2[int(edge_node)].reshape(-1)

                        
                degree = len(HyperGraph.node[node])
                num = degree ** (1 / node_num)
                node_emb =F.tanh((1 / math.factorial(node_num - 1)) * edge_emb) 
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
        return F.relu(fea_dim_tensor) + resdiual


class THNN_global_v1(nn.Module):
    def __init__(self, featdim, hiddendim, outputdim, n_class, rank = Rank, dropout = 0.3, global_degree = 1):
        super(THNN_global_v1, self).__init__()
        self.dropout = dropout
        self.model1 = THNN_global_layer( featdim, hiddendim, hiddendim, rank, dropout, global_degree)
        self.model2 = THNN_global_layer( outputdim, hiddendim, hiddendim, rank, dropout, global_degree)
        self.classifier = HGNN_classifier( hiddendim, n_class)
        self.global_embed1 = nn.Parameter(torch.rand(rank))
        self.register_parameter('global1',self.global_embed1)
        self.global_embed2 = nn.Parameter(torch.rand(rank))
        self.register_parameter('global2',self.global_embed2)


    def forward(self,HyperGraph, embedding):
        
        embedding = (embedding - embedding.mean())/(embedding.std())
        embedding_ = self.model1.forward(HyperGraph,  embedding, self.global_embed1)
        embedding_ = F.dropout(F.relu(embedding_), self.dropout)
        embedding_ = self.model2.forward(HyperGraph, embedding_, self.global_embed2)
        output = self.classifier(embedding_)
        return output