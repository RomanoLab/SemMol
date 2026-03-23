import torch
from torch_geometric.nn import HeteroConv,GATv2Conv,  Linear
import torch_geometric.transforms as T
import pickle
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score, brier_score_loss,average_precision_score, accuracy_score
import random
from tqdm import tqdm
import torch_geometric
import os
from torch_geometric.nn import MessagePassing, GINConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

import torch 
import pickle
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


import csv
import math
import time

import warnings
from typing import Dict, List, Optional

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops

from torch.utils.data.sampler import SubsetRandomSampler

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondDir
from rdkit.Chem.rdchem import BondStereo 
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  
from torch_geometric.data import HeteroData
from rdkit.Chem import BRICS


#### MolCLR
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, GINConv
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119 # valid: 1 to 118, including the extra mask tokens (0)
num_chirality_tag = 8 # 0 is the extra mask token
num_atom_degree = 12 # valid: 0 to 10, including the extra mask tokens (11)
num_atom_formal_charge = 15 # valid: -5 to 6 , including the extra mask tokens (7)
num_atom_hs = 10 # valid: 0 to 8 , including the extra mask tokens (9)
num_atom_radical_electrons = 6 # valid: 0 to 4 , including the extra mask tokens (5)
num_hybridization_type = 7 # 0 is the extra mask token
num_aromatic = 3 # valid: 0 to 1 , including the extra mask tokens (2)
num_ring = 3 # valid: 0 to 1 , including the extra mask tokens (2)

num_bond_type = 23 # including aromatic and self-loop edge (22)
num_stereo = 6
num_conjugated = 2


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_stereo, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_conjugated, emb_dim)
        
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:,0] = 22 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)


        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)


        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

    

            
class GINet_node_encoder(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, drop_ratio, 
                 num_layer=5, emb_dim=300):
        super(GINet_node_encoder, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = nn.Embedding(num_atom_degree, emb_dim)
        self.x_embedding4 = nn.Embedding(num_atom_formal_charge, emb_dim)
        self.x_embedding5 = nn.Embedding(num_atom_hs, emb_dim)
        self.x_embedding6 = nn.Embedding(num_atom_radical_electrons, emb_dim)
        self.x_embedding7 = nn.Embedding(num_hybridization_type, emb_dim)
        self.x_embedding8 = nn.Embedding(num_aromatic, emb_dim)
        self.x_embedding9 = nn.Embedding(num_ring, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        nn.init.xavier_uniform_(self.x_embedding9.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) + self.x_embedding7(x[:,6]) + self.x_embedding8(x[:,7]) + self.x_embedding9(x[:,8])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)    
        
        return h


    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


### HiMol
# num_atom_type = 121 # valid: 1 to 118, including the motif (120) and the graph (119) (NB there is also 0 to insert but not used)
# num_chirality_tag = 8 # 0 is the extra mask token
# num_atom_degree = 12 # valid: 0 to 10, including the extra mask tokens (11)
# num_atom_formal_charge = 15 # valid: -5 to 6 , including the extra mask tokens (9)
# num_atom_hs = 10 # valid: 0 to 8 , including the extra mask tokens (9)
# num_atom_radical_electrons = 9 # valid: 0 to 4 , including the extra mask tokens (5)
# num_hybridization_type = 7 # 0 is the extra mask token
# num_aromatic = 3 # valid: 0 to 1 , including the extra mask tokens (2)
# num_ring = 3 # valid: 0 to 1 , including the extra mask tokens (2)

# num_bond_type = 25 # including and self-loop edge (22), motif and graph edges
# num_stereo = 6
# num_conjugated = 2


# class GINEConv(MessagePassing):
#     def __init__(self, emb_dim):
#         super(GINEConv, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(emb_dim, 2*emb_dim), 
#             nn.ReLU(), 
#             nn.Linear(2*emb_dim, emb_dim)
#         )
#         self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
#         self.edge_embedding2 = nn.Embedding(num_stereo, emb_dim)
#         self.edge_embedding3 = nn.Embedding(num_conjugated, emb_dim)
#         nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
#         nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
#         nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

#     def forward(self, x, edge_index, edge_attr):
#         # add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

#         # add features corresponding to self-loop edges.
#         self_loop_attr = torch.zeros(x.size(0), 3)
#         self_loop_attr[:,0] = 22 #bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)


#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)


#         edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2])

#         return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

#     def message(self, x_j, edge_attr):
#         return x_j + edge_attr

#     def update(self, aggr_out):
#         return self.mlp(aggr_out)
    
    
# class GNN(torch.nn.Module):
#     """
    

#     Args:
#         num_layer (int): the number of GNN layers
#         emb_dim (int): dimensionality of embeddings
#         JK (str): last, concat, max or sum.
#         max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
#         drop_ratio (float): dropout rate
#         gnn_type: gin, gcn, graphsage, gat

#     Output:
#         node representations

#     """
#     def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
#         super(GNN, self).__init__()
#         self.num_layer = num_layer
#         self.drop_ratio = drop_ratio
#         self.JK = JK

#         if self.num_layer < 2:
#             raise ValueError("Number of GNN layers must be greater than 1.")

#         self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
#         self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
#         self.x_embedding3 = nn.Embedding(num_atom_degree, emb_dim)
#         self.x_embedding4 = nn.Embedding(num_atom_formal_charge, emb_dim)
#         self.x_embedding5 = nn.Embedding(num_atom_hs, emb_dim)
#         self.x_embedding6 = nn.Embedding(num_atom_radical_electrons, emb_dim)
#         self.x_embedding7 = nn.Embedding(num_hybridization_type, emb_dim)
#         self.x_embedding8 = nn.Embedding(num_aromatic, emb_dim)
#         self.x_embedding9 = nn.Embedding(num_ring, emb_dim)

#         nn.init.xavier_uniform_(self.x_embedding1.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding2.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding3.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding4.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding5.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding6.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding7.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding8.weight.data)
#         nn.init.xavier_uniform_(self.x_embedding9.weight.data)

#         ###List of MLPs
#         self.gnns = torch.nn.ModuleList()
#         for layer in range(num_layer):
#             if gnn_type == "gin":
#                 self.gnns.append(GINConv(emb_dim, aggr = "add"))
#             elif gnn_type == "gcn":
#                 self.gnns.append(GCNConv(emb_dim))
#             elif gnn_type == "gat":
#                 self.gnns.append(GATConv(emb_dim))
#             elif gnn_type == "graphsage":
#                 self.gnns.append(GraphSAGEConv(emb_dim))
#             elif gnn_type == "gine":
#                 self.gnns.append(GINEConv(emb_dim))

#         ###List of batchnorms
#         self.batch_norms = torch.nn.ModuleList()
#         for layer in range(num_layer):
#             self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

#     #def forward(self, x, edge_index, edge_attr):
#     def forward(self, *argv):
#         if len(argv) == 3:
#             x, edge_index, edge_attr = argv[0], argv[1], argv[2]
#         elif len(argv) == 1:
#             data = argv[0]
#             x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         else:
#             raise ValueError("unmatched number of arguments.")

#         x_h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) + self.x_embedding7(x[:,6]) + self.x_embedding8(x[:,7]) + self.x_embedding9(x[:,8])

#         h_list = [x_h]
#         for layer in range(self.num_layer):

#             h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
#             h = self.batch_norms[layer](h)
#             if layer == self.num_layer - 1:
#                 h = F.dropout(h, self.drop_ratio, training = self.training)
#             else:
#                 h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
#             h_list.append(h)

#         ### Different implementations of Jk-concat
#         if self.JK == "concat":
#             node_representation = torch.cat(h_list, dim = 1)
#         elif self.JK == "last":
#             node_representation = h_list[-1]
#         elif self.JK == "max":
#             h_list = [h.unsqueeze_(0) for h in h_list]
#             node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
#         elif self.JK == "sum":
#             h_list = [h.unsqueeze_(0) for h in h_list]
#             node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

#         return node_representation
    
    
# class GNN_graphpred(torch.nn.Module):
#     """
#     Extension of GIN to incorporate edge information by concatenation.

#     Args:
#         num_layer (int): the number of GNN layers
#         emb_dim (int): dimensionality of embeddings
#         drop_ratio (float): dropout rate
#         JK (str): last, concat, max or sum.
#         gnn_type: gin, gcn, graphsage, gat
        
#     """
#     def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
#         super(GNN_graphpred, self).__init__()
#         self.num_layer = num_layer
#         self.drop_ratio = drop_ratio
#         self.JK = JK
#         self.emb_dim = emb_dim

#         if self.num_layer < 2:
#             raise ValueError("Number of GNN layers must be greater than 1.")

#         self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        
#         if self.JK == "concat":
#             self.graph_pred_linear = torch.nn.Linear((self.num_layer + 1) * self.emb_dim)
#         else:

#             self.graph_pred_linear = torch.nn.Sequential(
#             torch.nn.Linear(self.emb_dim, (self.emb_dim)//2),
#             torch.nn.ELU(),
#             torch.nn.Linear((self.emb_dim)//2, (self.emb_dim)//2),
#             torch.nn.ELU(),
#             torch.nn.Linear((self.emb_dim)//2, 2)
#             )

#     def from_pretrained(self, model_file):
#         self.gnn.load_state_dict(torch.load(model_file))

#     def super_node_rep(self, node_rep):

#         return node_rep[-1,:]


#     def node_rep_(self, node_rep, batch):
#         super_group = []
#         batch_new = []
#         for i in range(len(batch)):
#             if i != (len(batch)-1) and batch[i] == batch[i+1]:
#                 super_group.append(node_rep[i,:])
#                 batch_new.append(batch[i].item())
#         super_rep = torch.stack(super_group, dim=0)
#         batch_new = torch.tensor(np.array(batch_new)).to(batch.device).to(batch.dtype)
#         return super_rep, batch_new

#     def mean_pool_(self, node_rep, batch):
#         super_group = [[] for i in range(32)]
#         for i in range(len(batch)):
#             super_group[batch[i]].append(node_rep[i,:])
#         node_rep = [torch.stack(list, dim=0).mean(dim=0) for list in super_group]
#         super_rep = torch.stack(node_rep, dim=0)
#         return super_rep

#     def node_group_(self, node_rep, batch):
#         super_group = [[] for i in range(32)]
#         for i in range(len(batch)):
#             super_group[batch[i]].append(node_rep[i,:])
#         node_rep = [torch.stack(list, dim=0) for list in super_group]
#         return node_rep 

#     def graph_emb(self, *argv):
#         if len(argv) == 4:
#             x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
#         elif len(argv) == 1:
#             data = argv[0]
#             x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         else:
#             raise ValueError("unmatched number of arguments.")

#         node_representation = self.gnn(x, edge_index, edge_attr)

#         super_rep = self.super_node_rep(node_representation)
#         return super_rep

#     def forward(self, *argv):
#         if len(argv) == 4:
#             x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
#         elif len(argv) == 1:
#             data = argv[0]
#             x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         elif len(argv) == 3:
#             x, edge_index, edge_attr = argv[0], argv[1], argv[2]
#         else:
#             raise ValueError("unmatched number of arguments.")

#         node_representation = self.gnn(x, edge_index, edge_attr)
        
#         super_rep = self.super_node_rep(node_representation)
        
#         return node_representation
    

class bottom_up_aggregator(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean') 
        
    def forward(self, x , edge_index):
    
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(0))
        
        out = self.propagate(edge_index, x=x)        
        out = out + x[1]
        
        return out
    
    def message(self, x_j):
        return x_j
    
def load_data_graph_dict(graph_str, if_CL, dict_gene_str,  dict_chem_str,  dict_assay_str,  dict_chemL_str, task_str):
    
    if if_CL:
        graph_str = graph_str.split('.pt')[0] + "_CL" + ".pt"
        
    Hetero_subgraph_comptoxAI = torch.load(graph_str)
    
    with open(dict_gene_str, "rb") as fp:
        node_name_numerical_idx_gene = pickle.load(fp)
    
    with open(dict_chem_str, "rb") as fp:
        node_name_numerical_idx_chemicals = pickle.load(fp)

    with open(dict_assay_str, "rb") as fp:
        node_name_numerical_idx_assay = pickle.load(fp)

    with open(dict_chemL_str, "rb") as fp:
        node_name_numerical_idx_chemlist = pickle.load(fp)

        
    if "pkl" in task_str:
        with open(task_str, "rb") as fp:
            list_task = pickle.load(fp)
            
    else:
        list_task = task_str
        
    return [Hetero_subgraph_comptoxAI, 
            node_name_numerical_idx_chemicals, 
            node_name_numerical_idx_gene, 
            node_name_numerical_idx_assay, 
            node_name_numerical_idx_chemlist, 
            list_task]
    


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    elif aggr == "cat":
        return torch.cat(xs, dim=-1)
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out
    
class HeteroConv_att(torch.nn.Module):

    def __init__(
        self,
        convs: Dict[EdgeType, MessagePassing],
        aggr: Optional[str] = "sum",
    ):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = {key[0] for key in convs.keys()}
        dst_node_types = {key[-1] for key in convs.keys()}
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        self.convs = ModuleDict(convs)
        self.aggr = aggr
        
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        *args_dict,
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[str, List[Tensor]] = {}
        attn_w_dict = {}

        for edge_type, conv in self.convs.items():
            src, rel, dst = edge_type

            has_edge_level_arg = False

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append((
                        value_dict.get(src, None),
                        value_dict.get(dst, None),
                    ))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                if not arg.endswith('_dict'):
                    raise ValueError(
                        f"Keyword arguments in '{self.__class__.__name__}' "
                        f"need to end with '_dict' (got '{arg}')")

                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (
                        value_dict.get(src, None),
                        value_dict.get(dst, None),
                    )

            if not has_edge_level_arg:
                continue

            out, attn_w = conv(*args, **kwargs, return_attention_weights=True)

            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)
                
            attn_w_dict[edge_type]=attn_w
                
        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict,attn_w_dict


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
    


class fancy_GNN(torch.nn.Module):
    def __init__(self, metadata, if_CL, hidden_channels, out_channels, num_layers, drop_ratio,norm_type,aggregation_type,
                molecular_embedder):
        super().__init__()
        
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.norm_type = norm_type
        
        self.molecular_encoder = molecular_embedder
        
        self.bottom_up = HeteroConv({
            ('Atom', 'hyper', 'Chemical'):bottom_up_aggregator()
                                }, aggr = None)
        
        # add for each edge a conv layer x number of layers
        self.convs = torch.nn.ModuleList()
        
        if if_CL:
        
            for _ in range(self.num_layers):
                conv = HeteroConv_att({

                     ('Chemical', 'CHEMICALHASACTIVEASSAY', 'Assay'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALHASINACTIVEASSAY', 'Assay'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALINCREASESEXPRESSION', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALDECREASESEXPRESSION', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALBINDSGENE', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Assay', 'ASSAYTARGETGENE', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'GENEINTERACTSWITHGENE', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Assay', 'rev_CHEMICALHASACTIVEASSAY', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Assay', 'rev_CHEMICALHASINACTIVEASSAY', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_CHEMICALINCREASESEXPRESSION', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_CHEMICALDECREASESEXPRESSION', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_CHEMICALBINDSGENE', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_ASSAYTARGETGENE', 'Assay'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('ChemicalList', 'LISTINCLUDESCHEMICAL', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'rev_LISTINCLUDESCHEMICAL', 'ChemicalList'):GATv2Conv(-1, hidden_channels,add_self_loops=False)

                 }, aggr=aggregation_type)
                self.convs.append(conv)

        else:
            for _ in range(self.num_layers):
                conv = HeteroConv_att({

                     ('Chemical', 'CHEMICALHASACTIVEASSAY', 'Assay'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALHASINACTIVEASSAY', 'Assay'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALINCREASESEXPRESSION', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALDECREASESEXPRESSION', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Chemical', 'CHEMICALBINDSGENE', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Assay', 'ASSAYTARGETGENE', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'GENEINTERACTSWITHGENE', 'Gene'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Assay', 'rev_CHEMICALHASACTIVEASSAY', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Assay', 'rev_CHEMICALHASINACTIVEASSAY', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_CHEMICALINCREASESEXPRESSION', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_CHEMICALDECREASESEXPRESSION', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_CHEMICALBINDSGENE', 'Chemical'):GATv2Conv(-1, hidden_channels,add_self_loops=False),
                     ('Gene', 'rev_ASSAYTARGETGENE', 'Assay'):GATv2Conv(-1, hidden_channels,add_self_loops=False)
                 }, aggr=aggregation_type)
                self.convs.append(conv)
            
        if self.norm_type=="batch":

            # add for each edge a batchnorm x number of layers
            self.batch_norms_chemical = torch.nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms_chemical.append(torch.nn.BatchNorm1d(hidden_channels))

            self.batch_norms_gene = torch.nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms_gene.append(torch.nn.BatchNorm1d(hidden_channels))

            self.batch_norms_assay = torch.nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms_assay.append(torch.nn.BatchNorm1d(hidden_channels))
                
        elif self.norm_type=="layer":
            # add for each edge a batchnorm x number of layers
            self.batch_norms_chemical = torch.nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms_chemical.append(torch.nn.LayerNorm(hidden_channels))

            self.batch_norms_gene = torch.nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms_gene.append(torch.nn.LayerNorm(hidden_channels))

            self.batch_norms_assay = torch.nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms_assay.append(torch.nn.LayerNorm(hidden_channels))
        
            
        # prediction layer for the chemicals
        self.pred_layer = Linear(hidden_channels, out_channels)
        
        
    def forward(self, x_dict, edge_index_list, edge_index_dict, chemical_label_task):
        
        ###### atom space within update

        # select the edge index of interest
        for edge_type in edge_index_list:
            if edge_type[0] == ('Atom', 'bond', 'Atom'):
                edge_index_molecules_space = edge_type[1]['edge_index']
                edge_attribute_molecules_space = edge_type[1]['edge_attr']
                break
        
            
        # embed atom with MolCLR embeding layers
        x_atom_encoded = self.molecular_encoder(x_dict['Atom'],edge_index_molecules_space, edge_attribute_molecules_space)
        x_dict['Atom'] = x_atom_encoded

        ####### bottom up aggregation (to create molcules)
        
        molecule_node_embeddings = self.bottom_up(x_dict,edge_index_dict)
        n_chemicals = molecule_node_embeddings['Chemical'].shape[0]
        d_chemicals = molecule_node_embeddings['Chemical'].shape[2]
        
        x_dict['Chemical'] = molecule_node_embeddings['Chemical'].reshape((n_chemicals,d_chemicals))
        
    
        ###### semantic space within update
        total_att = []
        for layer in range(self.num_layers):
            
            h,att = self.convs[layer](x_dict, edge_index_dict)
            #h = self.convs[layer](h)
            
            total_att.append(att)
            
            for node_type, x in h.items():
                
                if self.norm_type=="batch" or self.norm_type=="layer":
                
                    if node_type=="Chemical":
                        h[node_type] = self.batch_norms_chemical[layer](x)
                    elif node_type=="Gene":
                        h[node_type] = self.batch_norms_gene[layer](x)
                    elif node_type=="Assay":
                        h[node_type] = self.batch_norms_assay[layer](x)
                        
                    # chemicallist
                    elif node_type=="ChemicalList":
                        h[node_type] = self.batch_norms_assay[layer](x)


                if layer == self.num_layers - 1:
                    h[node_type] = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    h[node_type] = F.dropout(F.leaky_relu(x), self.drop_ratio, training=self.training)
            

        return self.pred_layer(h['Chemical'][chemical_label_task]),total_att
        
def run_semantic(assay_ML,graph_type, if_CL, results_fold, ckpt_fold, model_type):
    
    if model_type == "HiMol":
        model_folder_for_ckpt = './HiMol/finetune/saved_model/'
        model_folder_for_results = './HiMol/finetune/'
    else:
        model_folder_for_ckpt = './MolCLR/ckpt/comptoxai/GIN_node_masking/'
        model_folder_for_results = './MolCLR/'
    
    if len(assay_ML) !=1:
        if len(assay_ML) == 37:
            assay_repo = assay_ML[0].split("-")[0]
        else:
            assay_repo = assay_ML[0].split("_")[0]
    else:
        assay_repo = assay_ML[0]

    for init_seed in [127,128,129,130,131]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_epochs = 100
        
        # read the configuration of the best hyperparams for each model part
        with open("./" + model_type  + "/data/" + assay_repo  + "/best_hyperparams_MolCLR_5_runs.pkl", "rb") as fp:   
            best_hyperparams_MolCLR_5_runs = pickle.load(fp)

            
        if if_CL : 
            with open("./" + model_type + "/data/" + assay_repo + "/" + graph_type + "/best_hyperparams_semantic_CL_5_runs.pkl", "rb") as fp:   
                best_hyperparams_semantic_5_runs = pickle.load(fp)
        else:
            with open("./" + model_type + "/data/" + assay_repo + "/" + graph_type + "/best_hyperparams_semantic_5_runs.pkl", "rb") as fp:   
                best_hyperparams_semantic_5_runs = pickle.load(fp)

        import warnings
        warnings.filterwarnings("ignore")

        for assay in tqdm(list_task_tox21):


            hyperparams_MolCLR_finetuning = eval(best_hyperparams_MolCLR_5_runs[assay])
            hyperparams_semantic = eval(best_hyperparams_semantic_5_runs[assay])


    #         if os.path.exists("./MolCLR/results_tox21/semantic_and_graph_CL_" + str(init_seed) + "/test_" + assay + ".csv"):
    #             continue

            torch.manual_seed(init_seed)
            random.seed(init_seed)
            np.random.seed(init_seed)

            ######## loading dataset
            if assay == "hERG":
                dataset = pd.read_excel("./MolCLR/data/" + assay_repo + "/datasets_valid_and_splits/hGER_df.xlsx")
            else:
                dataset = pd.read_excel("./MolCLR/data/" + assay_repo + "/datasets_valid_and_splits/" + assay + "_df.xlsx")


            train_idx = np.load("./MolCLR/data/" + assay_repo + "/datasets_valid_and_splits/" + assay + "_train_idx_" + str(init_seed) + ".npy")
            valid_idx = np.load("./MolCLR/data/" + assay_repo + "/datasets_valid_and_splits/" + assay + "_valid_idx_" + str(init_seed) + ".npy")
            test_idx = np.load("./MolCLR/data/" + assay_repo + "/datasets_valid_and_splits/" + assay + "_test_idx_" + str(init_seed) + ".npy")

            training_dataset = dataset.loc[train_idx].reset_index(drop=True)
            validation_dataset = dataset.loc[valid_idx].reset_index(drop=True)
            test_dataset = dataset.loc[test_idx].reset_index(drop=True)

            ######## retrieve the assay idx and delete node and edges from the graph
            list_of_nodes_excluding_assay = []
            for key,value in node_name_numerical_idx_assay.items(): 
                if key!=assay:
                    list_of_nodes_excluding_assay.append(value)

            node_no_assay = torch.tensor(list_of_nodes_excluding_assay)

            # subgraph the graph according to the task of interest
            subset_dict = {
            'Chemicals': torch.tensor(list(node_name_numerical_idx_chemicals.values())),
            'Assay': node_no_assay,
            'Gene': torch.tensor(list(node_name_numerical_idx_gene.values())),
            'ChemicalList': torch.tensor(list(node_name_numerical_idx_chemlist.values()))
                }


            subgraph_Hetero_subgraph_comptoxAI = Hetero_subgraph_comptoxAI.subgraph(subset_dict)


            #preserve the edge index for the atom-atom (also edte attribute), and atom-chemical
            # -> we don't need direction 
            edge_index_bonds = subgraph_Hetero_subgraph_comptoxAI[(('Atom', 'bond', 'Atom'))]['edge_index']
            edge_attr_bonds = subgraph_Hetero_subgraph_comptoxAI[(('Atom', 'bond', 'Atom'))]['edge_attr']
            edge_index_hyper = subgraph_Hetero_subgraph_comptoxAI[(('Atom', 'hyper', 'Chemical'))]['edge_index']

            # delete these attributes of the graph 
            subgraph_Hetero_subgraph_comptoxAI[(('Atom', 'bond', 'Atom'))]['edge_index'] = None
            subgraph_Hetero_subgraph_comptoxAI[(('Atom', 'bond', 'Atom'))]['edge_attr'] = None
            subgraph_Hetero_subgraph_comptoxAI[(('Atom', 'hyper', 'Chemical'))]['edge_index'] = None

            #make undirected
            subgraph_Hetero_subgraph_comptoxAI = T.ToUndirected()(subgraph_Hetero_subgraph_comptoxAI)

            # reassing the information stored in the previous step
            subgraph_Hetero_subgraph_comptoxAI['Atom','hyper','Chemical'].edge_index = edge_index_hyper
            subgraph_Hetero_subgraph_comptoxAI['Atom','bond','Atom'].edge_index = edge_index_bonds
            subgraph_Hetero_subgraph_comptoxAI['Atom','bond','Atom'].edge_attr = edge_attr_bonds


            #########


            # add the label to the chemicals
            positive_chemicals = dataset[dataset['target']==1]['name'].values
            negative_chemicals = dataset[dataset['target']==0]['name'].values

            chemical_label_task = [] # -> True if edge in comptoxAI
            chemical_classification_label = [] # -> 1 if CHEMICALHASACTIVEASSAY , 0 if CHEMICALHASINACTIVEASSAY
            train_mask = []
            val_mask = []
            test_mask = []


            for chemicals in node_name_numerical_idx_chemicals.keys():
                if (chemicals in positive_chemicals) or (chemicals in negative_chemicals):

                    if (chemicals in positive_chemicals):
                        chemical_classification_label.append(1)
                    elif (chemicals in negative_chemicals):
                        chemical_classification_label.append(0)

                    chemical_label_task.append(True)
                else:
                    chemical_label_task.append(False)

                if chemicals in training_dataset['name'].values:
                    train_mask.append(True)
                    val_mask.append(False)
                    test_mask.append(False)
                elif chemicals in validation_dataset['name'].values:
                    train_mask.append(False)
                    val_mask.append(True)
                    test_mask.append(False)
                elif chemicals in test_dataset['name'].values:
                    train_mask.append(False)
                    val_mask.append(False)
                    test_mask.append(True)


            subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task = torch.tensor(chemical_label_task)
            subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label = torch.tensor(chemical_classification_label)
            subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask = torch.tensor(train_mask)
            subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask = torch.tensor(val_mask)
            subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask = torch.tensor(test_mask)
            subgraph_Hetero_subgraph_comptoxAI.to(device)



            class_weight = compute_class_weight('balanced',classes=np.unique(training_dataset['target']), y = training_dataset['target'])


            dataframe_results = pd.DataFrame(columns = ["hyperparameters_combination", "validation_wROCAUC",
                                                      "test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier",
                                                       "training_loss","validation_loss"]) 

        #     parameters_grid = {"hidden_channels":[128, 64],
        #                       "num_layers":[1, 2],
        #                        "drop_ratio_mol":[0, 0.3],
        #                       "drop_ratio_semantic":[0, 0.3],
        #                       "learning_rate_semantic":[0.01],
        #                       "weight_decay_semantic":[1e-2, 1e-3, 1e-4],
        #                       "norm":["","layer"],
        #                       "aggregation_node":["sum","mean"],
        #                       "weight_decay_molecular":['1e-6','1e-5'],
        #                       "init_base_lr":[5e-5, 5e-4]}

        #     grid_search = ParameterGrid(parameters_grid)



            ####### load the pretrained model
            if model_type == "HiMol":
                model_pretrained = GNN_graphpred(5, 300, "last", hyperparams_MolCLR_finetuning['drop_ratio'], 'gine')
                model_pretrained.from_pretrained("./HiMol/saved_model/pretrain.pth")
            else:
                model_pretrained = GINet_node_encoder(hyperparams_MolCLR_finetuning['drop_ratio'])
                model_pretrained.load_my_state_dict(torch.load('./MolCLR/ckpt/comptoxai/GIN_node_masking/model.pth',map_location='cuda:0'))
                
            # create the model
            model = fancy_GNN(subgraph_Hetero_subgraph_comptoxAI.metadata(),if_CL,
                              hidden_channels=hyperparams_semantic['hidden_channels'],
                              out_channels = 2,
                              num_layers = hyperparams_semantic['num_layers'],
                              drop_ratio= hyperparams_semantic['drop_ratio'],
                              norm_type=hyperparams_semantic['norm'],
                              aggregation_type=hyperparams_semantic['aggregation_node'],
                              molecular_embedder= model_pretrained).to(device)


            model.to(device)


            ## define optimizer with different learning rate and weight decay for the mol embedder and semantic model
            molecular_embedder_layer_list = []
            for name, param in model.named_parameters():
                if "molecular" in name:
                    molecular_embedder_layer_list.append(name)

            molecular_encoder_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in molecular_embedder_layer_list, model.named_parameters()))))
            semantic_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in molecular_embedder_layer_list, model.named_parameters()))))

            optimizer = torch.optim.Adam(
                        [{'params': molecular_encoder_params, 
                          'lr':hyperparams_MolCLR_finetuning['init_base_lr'] , 
                          'weight_decay':float(hyperparams_MolCLR_finetuning['weight_decay'])}, 
                         {'params': semantic_params}],  
                        lr=hyperparams_semantic['learning_rate'], weight_decay=float(hyperparams_semantic['weight_decay'])
                        )

            criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(class_weight, device=device, dtype=torch.float32))


            ### training loop

            best_val_metric = 0.0
            training_loss = []
            validation_loss = []

            for epoch in range(num_epochs):

                model.train()
                optimizer.zero_grad()

                # model forward
                out, att_wt = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
                            subgraph_Hetero_subgraph_comptoxAI.edge_items(),
                            subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                            subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task)

                
                # loss on the training set
                loss = criterion(out[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask],
                                subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask])

                validation_loss.append(loss.item())

                loss.backward()
                optimizer.step()

                # validation
                with torch.no_grad():
                    model.eval()
                    out, att_wv = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
                                subgraph_Hetero_subgraph_comptoxAI.edge_items(),
                                subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                                subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task)


                    loss = criterion(out[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask],
                                        subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask])

                    training_loss.append(loss.item())

                    prediction_prob = F.softmax(out,dim=1)
                    prediction = prediction_prob.argmax(dim=1)
                    val_auc = roc_auc_score(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask].cpu().detach().numpy(),
                                            prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask][:,1].cpu().detach().numpy())

                            
                    if val_auc>best_val_metric:
                        best_val_metric = val_auc
                        if if_CL:
                            if graph_type =="":
                                torch.save(model.state_dict(), './' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/semantic_graph_model_CL_test_' + assay + '.pth')
                            else:
                                torch.save(model.state_dict(), './' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/' + graph_type + '/semantic_graph_model_CL_test_' + assay + '.pth')
                        else:
                            if graph_type =="":
                                torch.save(model.state_dict(), './' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/semantic_graph_model_test_' + assay + '.pth')
                            else:
                                torch.save(model.state_dict(), './' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/' + graph_type + '/semantic_graph_model_test_' + assay + '.pth')


            with torch.no_grad(): 
                    
                if if_CL: 
                    if graph_type =="":
                        model.load_state_dict(torch.load('./' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/semantic_graph_model_CL_test_' + assay + '.pth', map_location=device))
                    else:
                        model.load_state_dict(torch.load('./' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/' + graph_type + '/semantic_graph_model_CL_test_' + assay + '.pth', map_location=device))
                else:
                    if graph_type =="":
                        model.load_state_dict(torch.load('./' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/semantic_graph_model_test_' + assay + '.pth', map_location=device))
                    else:
                        model.load_state_dict(torch.load('./' + model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_graphs_' + str(init_seed) + '/' + graph_type + '/semantic_graph_model_test_' + assay + '.pth', map_location=device))
                        
                    
                # test at the end of the training
                model.eval()
                out, att_wte = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
                            subgraph_Hetero_subgraph_comptoxAI.edge_items(),
                            subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                            subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task)



                prediction_prob = F.softmax(out,dim=1)
                prediction = prediction_prob.argmax(dim=1)
            # compute metrics
            test_auroc = roc_auc_score(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),
                                    prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask][:,1].cpu().detach().numpy())

            test_aupr = average_precision_score(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),
                                    prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask][:,1].cpu().detach().numpy())

            test_brier = brier_score_loss(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),
                                    prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask][:,1].cpu().detach().numpy())


            test_acc = accuracy_score(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),
                                    prediction[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy())

            test_mcc = matthews_corrcoef(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),
                                    prediction[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy())

            test_f1 = f1_score(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),
                                    prediction[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),average='weighted')

            new_row = pd.Series({"hyperparameters_combination":hyperparams_semantic,
                                "validation_wROCAUC":best_val_metric,

                                "test_wROCAUC":test_auroc,
                                "test_wPRAUC":test_aupr,
                                "test_brier":test_brier,
                                "test_mcc":test_mcc,
                                "test_wF1":test_f1,
                                "test_accuracy_score": test_acc,
                                "training_loss":training_loss,
                                "validation_loss":validation_loss})

            dataframe_results = pd.concat([dataframe_results,new_row.to_frame().T], ignore_index=True)

            if if_CL:
                if graph_type =="":
                    dataframe_results.to_csv("./" + model_folder_for_results + results_fold  + "/semantic_and_graph_CL_" + str(init_seed) + "/test_" + assay + ".csv",index=False)
                else:
                    dataframe_results.to_csv("./" + model_folder_for_results + results_fold + "/" + graph_type + "/semantic_and_graph_CL_" + str(init_seed) + "/test_" + assay + ".csv",index=False)
            else:
                if graph_type =="":
                    dataframe_results.to_csv("./" + model_folder_for_results + results_fold + "/semantic_and_graph_" + str(init_seed) + "/test_" + assay + ".csv",index=False)
                else:
                    dataframe_results.to_csv("./" + model_folder_for_results + results_fold + "/" + graph_type + "/semantic_and_graph_" + str(init_seed) + "/test_" + assay + ".csv",index=False)

                
def main():

    # tox21

    for if_CL in [False, True]:
        load_data_results = load_data_graph_dict("./MolCLR/data/Hetero_subgraph_comptoxAIs_hypernode.pt", if_CL, 
                            "./MolCLR/data/node_name_numerical_idx_gene.pkl",
                            "./MolCLR/data/node_name_numerical_idx_chemicals.pkl",
                            "./MolCLR/data/node_name_numerical_idx_assay.pkl",
                            "./MolCLR/data/node_name_numerical_idx_chemlist.pkl",
                            "./MolCLR/data/tox21/list_tasks_assay_ML.pkl")
            
        Hetero_subgraph_comptoxAI = load_data_results[0]
        node_name_numerical_idx_chemicals = load_data_results[1]
        node_name_numerical_idx_gene = load_data_results[2]
        node_name_numerical_idx_assay = load_data_results[3]
        node_name_numerical_idx_chemlist = load_data_results[4]
        list_task_tox21 = load_data_results[5]
                
        run_semantic(list_task_tox21, "", if_CL, "results_tox21", "tox21","MolCLR")
                
            
            

    # hERG
    list_task_tox21 = ['hERG']
    for if_CL in [False, True]:
        for assay in list_task_tox21:
            load_data_results = load_data_graph_dict("./MolCLR/data/" + assay  + "/Hetero_subgraph_comptoxAIs_hypernode.pt", if_CL, 
                                "./MolCLR/data/" + assay  + "/node_name_numerical_idx_gene.pkl",
                                "./MolCLR/data/" + assay + "/node_name_numerical_idx_chemicals.pkl",
                                "./MolCLR/data/" + assay  + "/node_name_numerical_idx_assay.pkl",
                                "./MolCLR/data/" + assay  + "/node_name_numerical_idx_chemlist.pkl",
                                ['hERG'])
            
            Hetero_subgraph_comptoxAI = load_data_results[0]
            node_name_numerical_idx_chemicals = load_data_results[1]
            node_name_numerical_idx_gene = load_data_results[2]
            node_name_numerical_idx_assay = load_data_results[3]
            node_name_numerical_idx_chemlist = load_data_results[4]

            run_semantic(list_task_tox21, "", if_CL, "results_herg", "herg","MolCLR")
            
    # neuro
    list_task_tox21 = ['neuro_BBB','neuro_NA','neuro_NC','neuro_NT']

    for if_CL in [False, True]:
        for assay in list_task_tox21:
            load_data_results = load_data_graph_dict("./MolCLR/data/neuro" + assay  + "/Hetero_subgraph_comptoxAIs_hypernode.pt", if_CL, 
                                    "./MolCLR/data/neuro" + assay  + "/node_name_numerical_idx_gene.pkl",
                                    "./MolCLR/data/neuro" + assay  + "/node_name_numerical_idx_chemicals.pkl",
                                    "./MolCLR/data/neuro"+ assay   + "/node_name_numerical_idx_assay.pkl",
                                    "./MolCLR/data/neuro"+ assay  + "/node_name_numerical_idx_chemlist.pkl",
                                    list_task_tox21)
                    
            Hetero_subgraph_comptoxAI = load_data_results[0]
            node_name_numerical_idx_chemicals = load_data_results[1]
            node_name_numerical_idx_gene = load_data_results[2]
            node_name_numerical_idx_assay = load_data_results[3]
            node_name_numerical_idx_chemlist = load_data_results[4]
                    
            run_semantic(list_task_tox21, "", if_CL, "results_neuro", "neuro","MolCLR")
            
        
    # CYP450   
    list_task_tox21 = ['CYP450_CYP1A2','CYP450_CYP2C9','CYP450_CYP2C19','CYP450_CYP2D6','CYP450_CYP3A4']

    for if_CL in [False, True]:
        for assay in list_task_tox21:
            load_data_results = load_data_graph_dict("./MolCLR/data/CYP450" + assay + "/Hetero_subgraph_comptoxAIs_hypernode.pt", if_CL, 
                                "./MolCLR/data/CYP450" + assay + "/node_name_numerical_idx_gene.pkl",
                                "./MolCLR/data/CYP450"  + assay + "/node_name_numerical_idx_chemicals.pkl",
                                "./MolCLR/data/CYP450" + assay + "/node_name_numerical_idx_assay.pkl",
                                "./MolCLR/data/CYP450"  + assay + "/node_name_numerical_idx_chemlist.pkl",
                                list_task_tox21)
                
            Hetero_subgraph_comptoxAI = load_data_results[0]
            node_name_numerical_idx_chemicals = load_data_results[1]
            node_name_numerical_idx_gene = load_data_results[2]
            node_name_numerical_idx_assay = load_data_results[3]
            node_name_numerical_idx_chemlist = load_data_results[4]
                
            run_semantic(list_task_tox21, "", if_CL, "results_CYP450", "CYP450","MolCLR")
                
    # Liver
    list_task_tox21 = ['Liver_2vs']

    for if_CL in [False, True]:
        for assay in list_task_tox21:
            load_data_results = load_data_graph_dict("./MolCLR/data/Liver" + assay + "/Hetero_subgraph_comptoxAIs_hypernode.pt", if_CL, 
                                "./MolCLR/data/Liver" + assay  + "/node_name_numerical_idx_gene.pkl",
                                "./MolCLR/data/Liver" + assay  + "/node_name_numerical_idx_chemicals.pkl",
                                "./MolCLR/data/Liver" + assay  + "/node_name_numerical_idx_assay.pkl",
                                "./MolCLR/data/Liver" + assay  + "/node_name_numerical_idx_chemlist.pkl",
                                list_task_tox21)
                
            Hetero_subgraph_comptoxAI = load_data_results[0]
            node_name_numerical_idx_chemicals = load_data_results[1]
            node_name_numerical_idx_gene = load_data_results[2]
            node_name_numerical_idx_assay = load_data_results[3]
            node_name_numerical_idx_chemlist = load_data_results[4]
                
            run_semantic(list_task_tox21, "", if_CL, "results_Liver", "Liver","MolCLR")
            
if __name__ == "__main__":
    main()
