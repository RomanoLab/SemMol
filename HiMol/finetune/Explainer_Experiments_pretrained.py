'''
Example (your custom pretrained GINE + LIG):
python Explainer_Experiments_pretrained.py --model_name CMPNN --attribution_name LIG --data_path "../MolRep/Datasets/Mutagenicity/CAS_N6512.csv" --dataset_name "Mutagenesis" --attribution_path "../MolRep/Datasets/Mutagenicity/attributions.npz" --smiles_col "SMILES" --target_col "label" --task_type Classification --multiclass_num_classes 1 --n_steps 128 --output_dir ../Outputs 
'''

import argparse
import os
from chemutils import get_mol, get_clique_mol
from rdkit.Chem import BRICS
import sys
from pathlib import Path
import pickle
from captum.attr import LayerIntegratedGradients
import yaml
import random
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
# MolCLR encoder
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from sklearn.utils.class_weight import compute_class_weight
import ast
import pickle
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
from torch_geometric.explain import GNNExplainer, CaptumExplainer,PGExplainer, Explainer
from torch_geometric.explain.metric import fidelity, characterization_score, fidelity_curve_auc,unfaithfulness
import torch_geometric
from captum.attr import LayerGradientXActivation
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv,GATv2Conv,Linear
import warnings
from typing import Dict, List, Optional

from torch import Tensor

from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops

num_atom_type = 121 # valid: 1 to 118, including the motif (120) and the graph (119) (NB there is also 0 to insert but not used)
num_chirality_tag = 8 # 0 is the extra mask token
num_atom_degree = 12 # valid: 0 to 10, including the extra mask tokens (11)
num_atom_formal_charge = 15 # valid: -5 to 6 , including the extra mask tokens (9)
num_atom_hs = 10 # valid: 0 to 8 , including the extra mask tokens (9)
num_atom_radical_electrons = 9 # valid: 0 to 4 , including the extra mask tokens (5)
num_hybridization_type = 7 # 0 is the extra mask token
num_aromatic = 3 # valid: 0 to 1 , including the extra mask tokens (2)
num_ring = 3 # valid: 0 to 1 , including the extra mask tokens (2)

num_bond_type = 25 # including and self-loop edge (22), motif and graph edges
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

        self.edge_emb_tap = nn.Identity()


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
        edge_embeddings = self.edge_emb_tap(edge_embeddings)

        self.last_edge_index = edge_index          # [2, E_used]
        self.last_num_nodes = x.size(0)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNN(torch.nn.Module):
    """
    

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = nn.Embedding(num_atom_degree, emb_dim)
        self.x_embedding4 = nn.Embedding(num_atom_formal_charge, emb_dim)
        self.x_embedding5 = nn.Embedding(num_atom_hs, emb_dim)
        self.x_embedding6 = nn.Embedding(num_atom_radical_electrons, emb_dim)
        self.x_embedding7 = nn.Embedding(num_hybridization_type, emb_dim)
        self.x_embedding8 = nn.Embedding(num_aromatic, emb_dim)
        self.x_embedding9 = nn.Embedding(num_ring, emb_dim)

        self.node_emb_tap_post = nn.Identity()

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)
        nn.init.xavier_uniform_(self.x_embedding7.weight.data)
        nn.init.xavier_uniform_(self.x_embedding8.weight.data)
        nn.init.xavier_uniform_(self.x_embedding9.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x_h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) + self.x_embedding7(x[:,6]) + self.x_embedding8(x[:,7]) + self.x_embedding9(x[:,8])

        h_list = [x_h]
        for layer in range(self.num_layer):

            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        node_representation = self.node_emb_tap_post(node_representation)
        return node_representation

class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        gnn_type: gin, gcn, graphsage, gat
        
    """
    def __init__(self, num_layer, emb_dim, semantic, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.semantic = semantic

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear((self.num_layer + 1) * self.emb_dim)
        else:

            self.graph_pred_linear = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, (self.emb_dim)//2),
            torch.nn.ELU(),
            torch.nn.Linear((self.emb_dim)//2, (self.emb_dim)//2),
            torch.nn.ELU(),
            torch.nn.Linear((self.emb_dim)//2, 2)
            )

    def from_pretrained(self, model_file, strict=True, map_location="cpu"):
        ckpt = torch.load(model_file, map_location=map_location)
        # Handle common wrappers
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        # Strip "module." if it was saved under DataParallel/DistributedDataParallel
        ckpt = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                for k, v in ckpt.items()}
        self.load_state_dict(ckpt, strict=strict)

    def super_node_rep(self, node_rep, batch):
        super_group = []
        for i in range(len(batch)):
            if i != (len(batch)-1) and batch[i] != batch[i+1]:
                super_group.append(node_rep[i,:])
            elif i == (len(batch) -1):
                super_group.append(node_rep[i,:])
        super_rep = torch.stack(super_group, dim=0)
        return super_rep


    def node_rep_(self, node_rep, batch):
        super_group = []
        batch_new = []
        for i in range(len(batch)):
            if i != (len(batch)-1) and batch[i] == batch[i+1]:
                super_group.append(node_rep[i,:])
                batch_new.append(batch[i].item())
        super_rep = torch.stack(super_group, dim=0)
        batch_new = torch.tensor(np.array(batch_new)).to(batch.device).to(batch.dtype)
        return super_rep, batch_new

    def mean_pool_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(len(batch)):
            super_group[batch[i]].append(node_rep[i,:])
        node_rep = [torch.stack(list, dim=0).mean(dim=0) for list in super_group]
        super_rep = torch.stack(node_rep, dim=0)
        return super_rep

    def node_group_(self, node_rep, batch):
        super_group = [[] for i in range(32)]
        for i in range(len(batch)):
            super_group[batch[i]].append(node_rep[i,:])
        node_rep = [torch.stack(list, dim=0) for list in super_group]
        return node_rep 

    def graph_emb(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        super_rep = self.super_node_rep(node_representation, batch)
        return super_rep

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        if (self.semantic == "gnn_encoder") or (self.semantic == "att"):
            return node_representation
        else:
            super_rep = self.super_node_rep(node_representation, batch)
            return self.graph_pred_linear(super_rep)



def read_smiles(datadf):
    smiles_data, labels, names = [], [], []
    smiles_transformed_as_a_graph = []
    for index, row in datadf.iterrows():

        smiles = row['smiles']
        label = row['target']
        name = row['name']
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and label != '':
            smiles_data.append(smiles)
            names.append(name)
            smiles_transformed_as_a_graph.append(index)
            labels.append(int(label))
    return smiles_data, labels, smiles_transformed_as_a_graph, names



def from_smiles_custom(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
    """Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """


    x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0,121)),
    'chirality': [
    'CHI_UNSPECIFIED',
    'CHI_TETRAHEDRAL_CW',
    'CHI_TETRAHEDRAL_CCW',
    'CHI_OTHER',
    'CHI_TETRAHEDRAL',
    'CHI_ALLENE',
    'CHI_SQUAREPLANAR',
    'CHI_TRIGONALBIPYRAMIDAL',
    'CHI_OCTAHEDRAL'
    ],
    'degree':
    list(range(0,12)),
    'formal_charge':
    list(range(-5,10)),
    'num_hs':
    list(range(0,9)),
    'num_radical_electrons':
    list(range(0,9)),
    'hybridization': [
    'UNSPECIFIED',
    'S',
    'SP',
    'SP2',
    'SP3',
    'SP3D',
    'SP3D2',
    'OTHER'
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
    }

    e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
    }


    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():  # type: ignore
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        self.smiles_data, self.labels, smiles_transformed_as_a_graph, self.names  = read_smiles(data_file)

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        smiles = self.smiles_data[idx]
        label = self.labels[idx]
        name = self.names[idx]
        mol_graph = MolGraph(smiles,label, name)  
        return mol_graph




class MolGraph(object):

    def __init__(self, smiles, label,name):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.molecule_data = from_smiles_custom(smiles)
        self.x_nosuper = self.molecule_data.x
        self.edge_index_nosuper = self.molecule_data.edge_index
        self.edge_attr_nosuper = self.molecule_data.edge_attr
        self.name = name

        # add label 
        self.y = torch.tensor(label, dtype=torch.long).view(1,-1)

        # add super node
        num_atoms = self.x_nosuper.size(0)
        super_x = torch.tensor([[119, 0, 11, 9, 9, 8, 0, 2, 2]]).to(self.x_nosuper.device)  

        #add motif 
        cliques = motif_decomp(self.mol)
        num_motif = len(cliques)
        if num_motif > 0:
            motif_x = torch.tensor([[120, 0, 11, 9, 9, 8, 0, 2, 2]]).repeat_interleave(num_motif, dim=0).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, motif_x, super_x), dim=0)

            motif_edge_index = []
            for k, motif in enumerate(cliques):
                motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
            motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)

            super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

            motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 3)
            motif_edge_attr[:,0] = 23 #bond type for motif
            motif_edge_attr = motif_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)

            super_edge_attr = torch.zeros(num_motif, 3)
            super_edge_attr[:,0] = 24 #bond type for super edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim = 0)

            self.num_part = (num_atoms, num_motif, 1)

        else:
            self.x = torch.cat((self.x_nosuper, super_x), dim=0)

            super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)

            super_edge_attr = torch.zeros(num_atoms, 3)
            super_edge_attr[:,0] = 24 #bond type for super edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, super_edge_attr), dim = 0)

            self.num_part = (num_atoms, 0, 1)

        self.batch = torch.zeros(self.x.shape[0])

    def size_node(self):
        return self.x.size()[0]

    def size_edge(self):
        return self.edge_attr.size()[0]

    def size_atom(self):
        return self.x_nosuper.size()[0]

    def size_bond(self):
        return self.edge_attr_nosuper.size()[0]


def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch


def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []  
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 


    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0: 
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms> len(c) > 0]


    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr)>1:
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
                    cliques[i] = list(set(cliques[i]) - set(list(ring)))
    
    cliques = [c for c in cliques if n_atoms> len(c) > 0]



    return cliques



class AtomsOnlyWrapper(nn.Module):
    """
    Wraps GNN_graphpred so forward() uses ATOMS-ONLY pooling (excludes super/motif),
    then reuses the same graph head. Submodules stay accessible for Captum taps.
    """
    def __init__(self, base: 'GNN_graphpred'):
        super().__init__()
        self.base = base  # keep original modules so Captum can hook base.gnn.node_emb_tap_post

    def forward(self, x, edge_index, edge_attr, batch):
        # 1) node embeddings from the same encoder (taps still inside self.base.gnn)
        H = self.base.gnn(x, edge_index, edge_attr)           # [N, D]
        B = int(batch.max().item()) + 1
        N, D = H.size(0), H.size(1)

        # sanity: batch must be length N
        assert batch.numel() == N, f"batch len {batch.numel()} != num nodes {N}"

        # 2) build atoms-only mask
        # heuristic default: drop the LAST node of each graph (super node)
        # find last index (end) per graph from batch
        # change points where graph id changes:
        change = (batch[1:] != batch[:-1])
        ends = torch.nonzero(change, as_tuple=False).squeeze(1) + 1
        ends = torch.cat([ends, torch.tensor([N-1], device=batch.device, dtype=torch.long)], dim=0)

        node_keep_mask = torch.ones(N, dtype=torch.bool, device=H.device)
        node_keep_mask[ends] = False  # drop super nodes

        # 3) masked mean pooling over atoms
        # sum of atom embeddings per graph
        h_graph = torch.zeros((B, D), device=H.device)
        H_masked = H * node_keep_mask.unsqueeze(1)            # zero-out non-atom nodes
        h_graph.index_add_(0, batch, H_masked)                # [B, D]

        # counts of kept nodes per graph (atoms only)
        counts = torch.bincount(batch[node_keep_mask], minlength=B).clamp(min=1).unsqueeze(1)  # [B, 1]
        h_graph = h_graph / counts

        # 4) reuse your graph head
        return self.base.graph_pred_linear(h_graph)

def atom_indices_from_batch(batch: torch.Tensor, atoms_per_graph: int = 29):
    N = batch.numel()
    # find graph boundaries
    change = (batch[1:] != batch[:-1])
    starts = torch.cat([torch.tensor([0], device=batch.device), torch.nonzero(change, as_tuple=False).squeeze(1) + 1])
    ends   = torch.cat([starts[1:] - 1, torch.tensor([N-1], device=batch.device)])  # inclusive

    idx_list = []
    for s, e in zip(starts.tolist(), ends.tolist()):
        # graph node order is [atoms ... motifs ... super]
        idx_list.append(torch.arange(s, s + atoms_per_graph, device=batch.device))
    return torch.cat(idx_list, dim=0)

def explain_atoms_only_layer_gxa(
    base_model,  # GNN_graphpred (has base_model.gnn.node_emb_tap_post and .gnn.gnns[k].edge_emb_tap)
    x, edge_index, edge_attr, batch,
    target,      # int or tensor([B])
    edge_layer_idx: int = 0,
    device='cpu',
    keep_node_idx=None,      # optional explicit atom indices (LongTensor)
    drop_self_loops: bool = True,
):
    model = AtomsOnlyWrapper(base_model).to(device).eval()

    x          = x.to(device).long()
    edge_index = edge_index.to(device).long()
    edge_attr  = edge_attr.to(device).long()
    batch      = batch.to(device).long()

    # --- Node tap (post-GNN, pre-pool) on the BASE encoder ---
    node_tap = model.base.gnn.node_emb_tap_post
    gxa_nodes = LayerGradientXActivation(model, node_tap)
    node_attr_emb = gxa_nodes.attribute(
        inputs=x,
        target=target,
        additional_forward_args=(edge_index, edge_attr, batch),
        attribute_to_layer_input=True
    )                           # [N, D]
    node_scores = node_attr_emb.abs().sum(-1)  # [N]

    # --- Edge tap inside chosen GINE layer on the BASE encoder ---
    edge_tap = model.base.gnn.gnns[edge_layer_idx].edge_emb_tap
    gxa_edges = LayerGradientXActivation(model, edge_tap)
    edge_attr_emb = gxa_edges.attribute(
        inputs=x,
        target=target,
        additional_forward_args=(edge_index, edge_attr, batch),
        attribute_to_layer_input=True
    )                           # [E_used, D]
    edge_scores = edge_attr_emb.abs().sum(-1)  # [E_used]

    # Align to exact edges the layer used:
    gine = model.base.gnn.gnns[edge_layer_idx]
    ei_used = gine.last_edge_index.to(device)  # set in forward of your GINEConv

    # ----- filter to atoms only -----
    N = node_scores.numel()
    if keep_node_idx is None:
        # same heuristic: drop last node per graph
        change = torch.where(torch.diff(batch, prepend=batch[:1]))[0]
        starts = torch.cat([torch.tensor([0], device=batch.device), change + 1], dim=0)
        ends   = torch.cat([starts[1:] - 1, torch.tensor([batch.numel()-1], device=batch.device)])
        atom_mask = torch.ones(N, dtype=torch.bool, device=device)
        atom_mask[ends] = False
        keep_node_idx = torch.where(atom_mask)[0]
    else:
        keep_node_idx = keep_node_idx.to(device)

    node_keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
    node_keep_mask[keep_node_idx] = True
    node_scores_atoms = node_scores[node_keep_mask].detach().cpu()
    node_attr_atoms   = node_attr_emb[node_keep_mask].detach().cpu()

    # ----- bonds only (drop edges touching super/motif; optionally drop self-loops) -----
    u, v = ei_used
    edge_keep = node_keep_mask[u] & node_keep_mask[v]
    if drop_self_loops:
        edge_keep = edge_keep & (u != v)

    edge_index_bonds  = ei_used[:, edge_keep].detach().cpu()
    edge_attr_bonds   = edge_attr_emb[edge_keep].detach().cpu()
    edge_scores_bonds = edge_scores[edge_keep].detach().cpu()

    return {
        "edge_index_used": ei_used.detach().cpu(),
        "node_attr_emb": node_attr_emb.detach().cpu(),
        "edge_attr_emb": edge_attr_emb.detach().cpu(),
        "node_scores": node_scores.detach().cpu(),
        "edge_scores": edge_scores.detach().cpu(),

        # filtered, atoms-only / bonds-only:
        "keep_node_idx": keep_node_idx.detach().cpu(),
        "node_attr_atoms": node_attr_atoms,
        "node_scores_atoms": node_scores_atoms,
        "edge_index_bonds": edge_index_bonds,
        "edge_attr_bonds": edge_attr_bonds,
        "edge_scores_bonds": edge_scores_bonds,
    }














def _unwrap_mol_gnn(m):
    enc = getattr(m, 'molecular_encoder', m)
    if hasattr(enc, 'module'): enc = enc.module        # DP/DDP
    if hasattr(enc, 'gnn'):    return enc.gnn          # GNN_graphpred → inner GNN
    return enc

# --- generic mover for nested structures (lists/tuples/dicts of tensors) ---
def _to_dev_nested(obj, dev):
    if obj is None: return None
    if torch.is_tensor(obj): return obj.to(dev)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dev_nested(x, dev) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_dev_nested(v, dev) for k, v in obj.items()}
    return obj

# --- proxy: Captum will perturb x_atom; we reassemble full hetero inputs for fancy_gnn ---
class _FancyProxy(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    # match fancy_GNN.forward(x_dict, edge_index_list, edge_index_dict, *extra)
    def forward(self, x_atom, x_dict_const, edge_index_list_const, edge_index_dict_const, *extra):
        x_dict = dict(x_dict_const)      # shallow copy ok
        x_dict['Atom'] = x_atom          # inject Captum-perturbed atoms
        return self.base(x_dict, edge_index_list_const, edge_index_dict_const, *extra)

def explain_molecular_encoder_gxa(
    fancy_model,                  # full fancy_gnn (semantic)
    x_dict, edge_index_dict, edge_index_list,   # full hetero inputs
    chem_idx,                     # int: Chemical index to explain
    target,                       # class id (int) or tensor [B]
    edge_layer_idx: int = 0,
    device: str = None,
    drop_self_loops: bool = True,
    *extra_args                   # any additional args your forward needs
):
    dev = device or next(fancy_model.parameters()).device
    proxy  = _FancyProxy(fancy_model).to(dev).eval()
    mol_gnn = _unwrap_mol_gnn(fancy_model)

    # move inputs
    x_dict_dev          = {k: (v.to(dev) if v is not None else None) for k, v in x_dict.items()}
    edge_index_dict_dev = {k: (v.to(dev) if v is not None else None) for k, v in edge_index_dict.items()}
    edge_index_list_dev = _to_dev_nested(edge_index_list, dev)
    x_atom = x_dict_dev['Atom'].long()

    # ---------- Node tap: post-GNN, pre-pool ----------
    node_tap  = mol_gnn.node_emb_tap_post
    gxa_nodes = LayerGradientXActivation(proxy, node_tap)
    node_attr_emb = gxa_nodes.attribute(
        inputs=x_atom,
        target=target,
        additional_forward_args=(x_dict_dev, edge_index_list_dev, edge_index_dict_dev, *extra_args),
        attribute_to_layer_input=True
    )                                  # [N_atoms_total, D]
    node_scores = node_attr_emb.abs().sum(-1)

    # ---------- Edge tap: GINE edge embedding input ----------
    assert 0 <= edge_layer_idx < len(mol_gnn.gnns), "edge_layer_idx out of range"
    edge_tap  = mol_gnn.gnns[edge_layer_idx].edge_emb_tap
    gxa_edges = LayerGradientXActivation(proxy, edge_tap)
    edge_attr_emb = gxa_edges.attribute(
        inputs=x_atom,
        target=target,
        additional_forward_args=(x_dict_dev, edge_index_list_dev, edge_index_dict_dev, *extra_args),
        attribute_to_layer_input=True
    )                                 # [E_used_by_layer, D]
    edge_scores = edge_attr_emb.abs().sum(-1)

    # exact edges used by tapped layer (after its self-loops)
    gine    = mol_gnn.gnns[edge_layer_idx]
    ei_used = gine.last_edge_index.to(dev)          # [2, E_used]

    # ---------- Molecule-only filtering ----------
    A_C = edge_index_dict_dev[('Atom','hyper','Chemical')]    # [2, num_hyper]
    atom_ids = A_C[0, A_C[1] == int(chem_idx)].unique()

    node_scores_atoms = node_scores[atom_ids].detach().cpu()
    node_attr_atoms   = node_attr_emb[atom_ids].detach().cpu()

    atom_mask = torch.zeros(x_atom.size(0), dtype=torch.bool, device=dev)
    atom_mask[atom_ids] = True
    u, v = ei_used
    keep = atom_mask[u] & atom_mask[v]
    if drop_self_loops:
        keep = keep & (u != v)

    edge_index_bonds  = ei_used[:, keep].detach().cpu()
    edge_attr_bonds   = edge_attr_emb[keep].detach().cpu()
    edge_scores_bonds = edge_scores[keep].detach().cpu()

    return {
        "edge_index_used": ei_used.detach().cpu(),
        "node_attr_emb":   node_attr_emb.detach().cpu(),
        "edge_attr_emb":   edge_attr_emb.detach().cpu(),
        "node_scores":     node_scores.detach().cpu(),
        "edge_scores":     edge_scores.detach().cpu(),

        "atom_ids":            atom_ids.detach().cpu(),
        "node_attr_atoms":     node_attr_atoms,
        "node_scores_atoms":   node_scores_atoms,
        "edge_index_bonds":    edge_index_bonds,
        "edge_attr_bonds":     edge_attr_bonds,
        "edge_scores_bonds":   edge_scores_bonds,
    }


##########################

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
        
        
    def forward(self, x_dict, edge_index_list, edge_index_dict):
        
        ###### atom space within update

        # select the edge index of interest
        for edge_type in edge_index_list:
            if edge_type[0] == ('Atom', 'bond', 'Atom'):
                edge_index_molecules_space = edge_type[1]['edge_index']
                edge_attribute_molecules_space = edge_type[1]['edge_attr']
                break
        
        # embed atom with MolCLR embeding layers
        x_atom_encoded = self.molecular_encoder(x_dict['Atom'],edge_index_molecules_space, edge_attribute_molecules_space, "")
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

        if self.molecular_encoder.semantic == "gnn_encoder":
            return self.pred_layer(h['Chemical'])
        elif self.molecular_encoder.semantic == "att":
            return self.pred_layer(h['Chemical']),total_att

class FancyVectorWrapper(nn.Module):
    """
    Returns logits for ONE chemical as a 2D tensor [1, C] so Captum can select a target column.
    Accepts (x_dict, edge_index_dict) because PyG's Captum wrapper calls it that way.
    """
    def __init__(self, base_model, chem_idx: int, edge_index_list, edge_index_dict):
        super().__init__()
        self.base = base_model
        self.chem_idx = int(chem_idx)
        self.edge_index_list = edge_index_list
        self.edge_index_dict = edge_index_dict

    def forward(self, x_dict, edge_index_dict=None, *args, **kwargs):
        eid_dict = edge_index_dict if edge_index_dict is not None else self.edge_index_dict
        logits_all = self.base(
            x_dict,
            self.edge_index_list,
            eid_dict,
            chemical_label_task=None      # must return [num_chem, C]
        )
        vec = logits_all[self.chem_idx]    # [C]
        if vec.dim() == 1:                 # ensure batch dimension
            vec = vec.unsqueeze(0)         # [1, C]
        return vec



def undirected_edge_scores(edge_index, scores, num_nodes=None):
    u, v = edge_index
    if num_nodes is None:
        num_nodes = int(torch.max(torch.stack([u.max(), v.max()])).item()) + 1
    lo = torch.minimum(u, v)
    hi = torch.maximum(u, v)
    key = (lo.long() * num_nodes) + hi.long()
    uniq, inv = torch.unique(key, return_inverse=True)
    agg = torch.zeros_like(uniq, dtype=scores.dtype)
    agg.index_add_(0, inv, scores)
    lo_uniq = (uniq // num_nodes).long()
    hi_uniq = (uniq %  num_nodes).long()
    undirected_ei = torch.stack([lo_uniq, hi_uniq], dim=0)
    return undirected_ei, agg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Explain GNN encoder prediction')
    parser.add_argument('--semantic', type=str, default="",
                        help='if "" use only gnn encoder, if "gnn_encoder" use semantic only for that, and if att check attention weights')
    parser.add_argument('--CL', type=int, default=0,
                        help='if CL consider the chemical list node in the semantic encoder')

    args = parser.parse_args()

    # MolCLR
    config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    if torch.cuda.is_available() and config['gpu'] != 'cpu':
        device = config['gpu']
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    config['dataset']['task'] = 'classification'
    config['dataset']['target'] = 1
    
    list_tasks = ['Liver_2vs']
    
    with open("../data/Liver/best_hyperparams_MolCLR_5_runs.pkl", "rb") as fp:   
            best_hyperparams_MolCLR_5_runs = pickle.load(fp)
            

    # Semantic
    if args.semantic == "gnn_encoder":

        load_data_results = load_data_graph_dict("../../MolCLR/data/Liver/alone/Hetero_subgraph_comptoxAIs_hypernode.pt", args.CL == 1, 
                                "../../MolCLR/data/Liver/alone/node_name_numerical_idx_gene.pkl",
                                "../../MolCLR/data/Liver/alone/node_name_numerical_idx_chemicals.pkl",
                                "../../MolCLR/data/Liver/alone/node_name_numerical_idx_assay.pkl",
                                 "../../MolCLR/data/Liver/alone/node_name_numerical_idx_chemlist.pkl",
                                ['Liver_2vs'])

        Hetero_subgraph_comptoxAI = load_data_results[0]
        node_name_numerical_idx_chemicals = load_data_results[1]
        node_name_numerical_idx_gene = load_data_results[2]
        node_name_numerical_idx_assay = load_data_results[3]
        node_name_numerical_idx_chemlist = load_data_results[4]


        if args.CL == 0: 
            with open("../data/Liver/alone/best_hyperparams_semantic_5_runs.pkl", "rb") as fp:   
                best_hyperparams_semantic_5_runs = pickle.load(fp)
        else:
            with open("../data/Liver/alone/best_hyperparams_semantic_CL_5_runs.pkl", "rb") as fp:   
                best_hyperparams_semantic_5_runs = pickle.load(fp)

        warnings.filterwarnings("ignore")

    for init_seed in [127,128,129,130,131]:

        torch.manual_seed(init_seed)
        random.seed(init_seed)
        np.random.seed(init_seed)

        for num_task, task_name in enumerate(list_tasks):

            if init_seed == 127:
                parameters_combination = best_hyperparams_MolCLR_5_runs[task_name]
                best_hyperparams_MolCLR_5_runs = eval(best_hyperparams_MolCLR_5_runs[task_name])
                best_hyperparams_MolCLR_5_runs['parameters_combination'] = parameters_combination
                

            config['seed'] = init_seed

            torch.manual_seed(init_seed)
            random.seed(init_seed)
            np.random.seed(init_seed)

            datadf = pd.read_excel("../data/Liver/datasets_valid_and_splits/" + task_name + "_df.xlsx")
                
            config['init_lr'] = best_hyperparams_MolCLR_5_runs['init_lr']
            config['init_base_lr'] = best_hyperparams_MolCLR_5_runs['init_base_lr']
            config['pred_n_layer'] = best_hyperparams_MolCLR_5_runs['pred_n_layer']
            config['weight_decay'] = best_hyperparams_MolCLR_5_runs['weight_decay']
            config['drop_ratio'] = best_hyperparams_MolCLR_5_runs['drop_ratio']
            config['feat_dim'] = best_hyperparams_MolCLR_5_runs['feat_dim']


            datadf_pos = datadf[datadf['target'] == 1]

            attribution_df = pd.read_excel("../data/Liver/Liver_.xlsx")
            attribution_df['name'] = attribution_df['Drug Name']
            attribution_df['smiles'] = attribution_df['n.sMILES']

            new_df = datadf_pos.merge(attribution_df[['name','smiles','attribution']])
            all_df = datadf.merge(attribution_df[['name','smiles','attribution']])

            molecule_dataset_positive = MoleculeDataset(new_df)
            all_dataset = MoleculeDataset(all_df)

            if args.semantic == "":

                model = GNN_graphpred(config['model']['num_layer'], config['model']['emb_dim'], args.semantic, JK = "last", drop_ratio = config['drop_ratio'], gnn_type = 'gine')
                    
                model.from_pretrained("./saved_model/Liver/best_run_models_" + str(init_seed)  + "/" + task_name + "_model_test.pth",strict=True, map_location="cpu")
        
                model.eval().to('cpu')


            else:
                model_pretrained = GNN_graphpred(5, 300, args.semantic, "last", best_hyperparams_MolCLR_5_runs['drop_ratio'], 'gine')

                if init_seed == 127:
                    parameters_combination_sem = best_hyperparams_semantic_5_runs[task_name]
                    hyperparams_semantic = eval(best_hyperparams_semantic_5_runs[task_name])
                    hyperparams_semantic['parameters_combination'] = parameters_combination_sem

                ######## retrieve the assay idx and delete node and edges from the graph
                list_of_nodes_excluding_assay = []
                for key,value in node_name_numerical_idx_assay.items(): 
                    if key!=task_name:
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
                positive_chemicals = datadf[datadf['target']==1]['name'].values
                negative_chemicals = datadf[datadf['target']==0]['name'].values

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

                subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task = torch.tensor(chemical_label_task)
                subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label = torch.tensor(chemical_classification_label)
                subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask = torch.tensor(train_mask)
                subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask = torch.tensor(val_mask)
                subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask = torch.tensor(test_mask)
                subgraph_Hetero_subgraph_comptoxAI.to('cpu')

                model = fancy_GNN(subgraph_Hetero_subgraph_comptoxAI.metadata(),args.CL == 1,
                              hidden_channels=hyperparams_semantic['hidden_channels'],
                              out_channels = 2,
                              num_layers = hyperparams_semantic['num_layers'],
                              drop_ratio= hyperparams_semantic['drop_ratio'],
                              norm_type=hyperparams_semantic['norm'],
                              aggregation_type=hyperparams_semantic['aggregation_node'],
                              molecular_embedder= model_pretrained).to(device)


                if args.CL == 1:
                    semmol = torch.load('./saved_model/Liver/best_run_semantics_graphs_' + str(init_seed) + "/alone/semantic_graph_model_CL_test_" + task_name + ".pth")
                else:
                    semmol = torch.load('./saved_model/Liver/best_run_semantics_graphs_' + str(init_seed) + "/alone/semantic_graph_model_test_" + task_name + ".pth")
                

                model.load_state_dict(semmol)
                model.eval().to('cpu')


            # methods = [ "Layer_Saliency",
            #     CaptumExplainer(attribution_method="Saliency"),   
            # CaptumExplainer(attribution_method="InputXGradient"),    
            # GNNExplainer(lr=0.01),      
            # GNNExplainer(lr=0.001),
            # PGExplainer(epochs=100, lr=0.001),
            # PGExplainer(epochs=100, lr=0.01)]

            # methods_n = ['Layer_Saliency','Saliency','InputXGradient','GNNExp_01','GNNExp_001','PGExp_01','PGExp_001']
   

            for explain_method,explainer_name in zip(['Layer_Saliency'],['Layer_Saliency']):

                # GNN encoder finetuned
                if args.semantic == "":

                    if explainer_name == "Layer_Saliency":
                        for molecule_data in tqdm(molecule_dataset_positive):

                            keep_node_idx = atom_indices_from_batch(molecule_data.batch, atoms_per_graph=molecule_data.x_nosuper.shape[0])
                            res = explain_atoms_only_layer_gxa(
                                                            base_model=model,
                                                            x=molecule_data.x, edge_index=molecule_data.edge_index, edge_attr=molecule_data.edge_attr, batch=molecule_data.batch,
                                                            target=1,
                                                            edge_layer_idx=4,
                                                            device='cpu',
                                                            keep_node_idx=keep_node_idx,
                                                            drop_self_loops=True
                                                        )


                            new_res = {'node_scores':res['node_scores_atoms'], 'edge_scores':res['edge_scores_bonds'], 'bond_idx':res['edge_index_bonds']}

                            with open("./results_Liver/explain/" + explainer_name + "/" + str(init_seed) + "/" + task_name + "/" + molecule_data.name + ".pkl", "wb") as f:
                                pickle.dump(new_res, f)


                    else:
                        explainer = Explainer(
                                        model=model,
                                        algorithm=explain_method,
                                        explanation_type='phenomenon',
                                        node_mask_type=None,
                                        edge_mask_type='object',

                                        model_config=dict(
                                            mode='multiclass_classification',
                                            task_level='graph',
                                            return_type='raw',
                                        ),
                                        )
                        
                        if "PGExp" in explainer_name:
                            for epoch in range(100):
                                for index,molecule_data in enumerate(all_dataset):  # Indices to train against.
                                    loss = explainer.algorithm.train(epoch = epoch, model = model.to('cpu'), x = molecule_data.x, edge_index = molecule_data.edge_index,
                                                                        target=molecule_data.y.flatten(), index=0,  attr=molecule_data.edge_attr, x_batch=molecule_data.batch )
                            

                        for molecule_data in tqdm(molecule_dataset_positive):

                            explanation_compound = explainer(x = molecule_data.x, edge_index = molecule_data.edge_index, attr=molecule_data.edge_attr, target = molecule_data.y.flatten(),x_batch=molecule_data.batch,index=0)

                            torch.save(explanation_compound,"./results_Liver/explain/" + explainer_name + "/" + str(init_seed) + "/" + task_name + "/" + molecule_data.name + ".pth")

                            sys.exit()
                            # print(fidelity(explainer,explanation_compound))
                            # pos_fid, neg_fid = fidelity(explainer,explanation_compound)
                            # print(characterization_score(pos_fid,neg_fid))
                            # ks = torch.linspace(1/steps, 1.0, steps)
                            # xs = torch.tensor([frac.item() for frac in ks])
                            # print(fidelity_curve_auc(pos_fid,neg_fid,xs))
                            # print(unfaithfulness(explainer,explanation_compound,10))
                            # sys.exit()

                        torch.save(explainer.algorithm.state_dict(),"./results_Liver/explain/" + explainer_name + "/" + str(init_seed) + "/" + task_name + "/model_exp.pt")

                # GNN encoder + semantic GNN
                else:
                    if explainer_name == "Layer_Saliency":

                        for molecule_data in tqdm(molecule_dataset_positive):
                            chemical_idx = node_name_numerical_idx_chemicals[molecule_data.name]

                            res = explain_molecular_encoder_gxa(
                                                                model,
                                                                x_dict=subgraph_Hetero_subgraph_comptoxAI.x_dict,
                                                                edge_index_list=list(subgraph_Hetero_subgraph_comptoxAI.edge_items()),  # ok
                                                                edge_index_dict=subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                                                                chem_idx=chemical_idx,
                                                                target=1,
                                                                edge_layer_idx=4,
                                                                device='cpu'
                                                            )

                            new_res = {'node_scores':res['node_scores_atoms'], 'atom_idx_in_hetero':res['node_scores_atoms'],'edge_scores':res['edge_scores_bonds'], 'bond_idx_in_hetero':res['edge_index_bonds']}
                            
                            if args.CL == 0:

                                with open("./results_Liver/explain_sem/" + explainer_name + "/" + str(init_seed) + "/" + task_name + "/" + molecule_data.name + ".pkl", "wb") as f:
                                    pickle.dump(new_res, f)

                            else:

                                with open("./results_Liver/explain_sem_CL/" + explainer_name + "/" + str(init_seed) + "/" + task_name + "/" + molecule_data.name + ".pkl", "wb") as f:
                                    pickle.dump(new_res, f)
           


                        # scalar_model = FancyVectorWrapper(
                        #         base_model=model,
                        #         chem_idx=chemical_idx,
                        #         edge_index_list=subgraph_Hetero_subgraph_comptoxAI.edge_items(),
                        #         edge_index_dict=subgraph_Hetero_subgraph_comptoxAI.edge_index_dict
                        #     ).eval()

                        # explainer = Explainer(
                        #     model=scalar_model,
                        #     algorithm=explain_method,
                        #     explanation_type='phenomenon',
                        #     node_mask_type=None,
                        #     edge_mask_type='object',
                        #     model_config=dict(
                        #         mode='multiclass_classification',
                        #         task_level='graph',
                        #         return_type='raw',
                        #     ),
                        # )

                        # explanation = explainer( x=subgraph_Hetero_subgraph_comptoxAI.x_dict,
                        #                         edge_index=subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                        #                         target = 1
                        #                     )

                        # print(explanation)
                        # sys.exit()


