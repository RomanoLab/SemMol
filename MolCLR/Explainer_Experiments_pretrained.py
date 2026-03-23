'''
Example (your custom pretrained GINE + LIG):
python Explainer_Experiments_pretrained.py --model_name CMPNN --attribution_name LIG --data_path "../MolRep/Datasets/Mutagenicity/CAS_N6512.csv" --dataset_name "Mutagenesis" --attribution_path "../MolRep/Datasets/Mutagenicity/attributions.npz" --smiles_col "SMILES" --target_col "label" --task_type Classification --multiclass_num_classes 1 --n_steps 128 --output_dir ../Outputs 
'''

import argparse
import os
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
from sklearn.manifold import TSNE
from torch import Tensor

from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops

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

        self.edge_emb_tap = nn.Identity()

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

    def forward(self, x, edge_index, edge_attr):

        N = x.size(0)

        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:,0] = 22 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2])
        edge_embeddings = self.edge_emb_tap(edge_embeddings)

        self.last_edge_index = edge_index          # shape [2, E_used_by_layer]
        self.last_num_nodes = x.size(0)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, pred_n_layer,drop_ratio,feat_dim, semantic,
        task='classification', num_layer=5, emb_dim=300, 
        pool='mean', pred_act='softplus'
    ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = 'classification'
        self.semantic = semantic

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

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if self.semantic == "":

            if pool == 'mean':
                self.pool = global_mean_pool
            elif pool == 'max':
                self.pool = global_max_pool
            elif pool == 'add':
                self.pool = global_add_pool
            
            # feat_lin if using only the graph embeddings
            self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)


            if self.task == 'classification':
                out_dim = 2
            elif self.task == 'regression':
                out_dim = 1
            
            self.pred_n_layer = max(1, pred_n_layer)

            if pred_act == 'relu':
                pred_head = [
                    nn.Linear(self.feat_dim, self.feat_dim//2), 
                    nn.ReLU(inplace=True)
                ]
                for _ in range(self.pred_n_layer - 1):
                    pred_head.extend([
                        nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                        nn.ReLU(inplace=True),
                    ])
                pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
            elif pred_act == 'softplus':
                pred_head = [
                    nn.Linear(self.feat_dim, self.feat_dim//2), 
                    nn.Softplus()
                ]
                for _ in range(self.pred_n_layer - 1):
                    pred_head.extend([
                        nn.Linear(self.feat_dim//2, self.feat_dim//2), 
                        nn.Softplus()
                    ])
            else:
                raise ValueError('Undefined activation function')
        
            pred_head.append(nn.Linear(self.feat_dim//2, out_dim))
            self.pred_head = nn.Sequential(*pred_head)

    def _embed_nodes(self, x):

        x = x.long()
        h = ( self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1]) +
              self.x_embedding3(x[:, 2]) + self.x_embedding4(x[:, 3]) +
              self.x_embedding5(x[:, 4]) + self.x_embedding6(x[:, 5]) +
              self.x_embedding7(x[:, 6]) + self.x_embedding8(x[:, 7]) +
              self.x_embedding9(x[:, 8]) )
        return h  # [N, D]

    def forward(self, x, edge_index, attr, x_batch):

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) + self.x_embedding7(x[:,6]) + self.x_embedding8(x[:,7]) + self.x_embedding9(x[:,8])

        # --- embed nodes, then pass through tap (for Captum layer methods)
        h = self._embed_nodes(x)            # [N, D]
        #h = self.node_emb_tap(h)            

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        if (self.semantic == "gnn_encoder") or (self.semantic == "att"):
            h = self.node_emb_tap_post(h)
            return h
        else:
            h = self.node_emb_tap_post(h)
            h = self.pool(h, batch=x_batch)

            ## following row for embeddings 
            #return h
            

            ## following rows for explainability
            h = self.feat_lin(h)

            return self.pred_head(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)



ATOM_LIST = list(range(0,119))
CHIRALITY_LIST = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
    ChiralType.CHI_TETRAHEDRAL,
    ChiralType.CHI_ALLENE,
    ChiralType.CHI_SQUAREPLANAR,
    ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    ChiralType.CHI_OCTAHEDRAL
]
DEGREE_LIST = list(range(0,12))
FORMAL_CHARGE_LIST = list(range(-5,8))
NUM_HS_LIST = list(range(0,9))
NUM_RADICAL_ELECTRONS_LIST = list(range(0,6))
HYBRIDIZATION_LIST = [
    HybridizationType.UNSPECIFIED,
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.OTHER
]
AROMATIC_LIST = list(range(0,3))
RING_LIST = list(range(0,3))


BOND_LIST = [
    BondType.UNSPECIFIED, 
    BondType.SINGLE, 
    BondType.DOUBLE, 
    BondType.TRIPLE, 
    BondType.QUADRUPLE,
    BondType.QUINTUPLE,
    BondType.HEXTUPLE,
    BondType.ONEANDAHALF,
    BondType.TWOANDAHALF,
    BondType.THREEANDAHALF,
    BondType.FOURANDAHALF,
    BondType.FIVEANDAHALF,
    BondType.AROMATIC,
    BondType.IONIC,
    BondType.HYDROGEN,
    BondType.THREECENTER,
    BondType.DATIVEONE,
    BondType.DATIVE,
    BondType.DATIVEL,
    BondType.DATIVER,
    BondType.OTHER,
    BondType.ZERO
]

STEREO_LIST = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS
]

CONJUGATED_LIST = list(range(0,2))


def read_smiles(datadf, target, task):
    smiles_data, labels, attributions, names = [], [], [], []
    smiles_transformed_as_a_graph = []
    for index, row in datadf.iterrows():

        smiles = row['smiles']
        label = row['target']
        name = row['name']
        attribution = np.array(ast.literal_eval(row['attribution'].replace(".",",")))   
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and label != '':
            smiles_data.append(smiles)
            names.append(name)
            smiles_transformed_as_a_graph.append(index)
            attributions.append(attribution)
            if task == 'classification':
                labels.append(int(label))
            elif task == 'regression':
                labels.append(float(label))
            else:
                ValueError('task must be either regression or classification')

    return smiles_data, labels, attributions, smiles_transformed_as_a_graph, names



def from_smiles_custom(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
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
    list(range(0,119)),
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
    list(range(0,6)),
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

class MolTestDataset(Dataset):
    def __init__(self, datadf, target, task, task_name):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels, self.attributions, smiles_transformed_as_a_graph, self.names = read_smiles(datadf, target, task)
        self.task = task

        # filter the dataframe according to the smiles to transformed to graph by rdkit
        self.filtered_datadf = datadf.filter(items=smiles_transformed_as_a_graph, axis=0).reset_index(drop=True)

        #self.filtered_datadf.to_excel("./data/datasets_valid_and_splits/" + task_name + ".xlsx",index=False)
        self.conversion = 1

    def __getitem__(self, index):
        mol_graph_pyg = from_smiles_custom(self.smiles_data[index])
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        #mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        M_f = mol_graph_pyg.edge_attr.shape[1]

        x = mol_graph_pyg.x

        edge_index = mol_graph_pyg.edge_index
        edge_attr = mol_graph_pyg.edge_attr
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1) 
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, attribution = torch.tensor(self.attributions[index]), name = self.names[index], smile = self.smiles_data[index],batch=torch.zeros(x.shape[0],dtype=torch.long))

        return data

    def __len__(self):
        return len(self.smiles_data)


@torch.no_grad()
def _to_dev(x, dev):
    return None if x is None else x.to(dev)

def layer_gxa_node_and_edge_post(
    model,                     # GINet with .node_emb_tap_post and GINEConv.edge_emb_tap
    x, edge_index, attr, x_batch,
    target,                   # class index (int) or tensor of shape [B] / scalar
    edge_layer_idx: int = 0,  # which GINE layer’s edges to target
    device=None
):
    """
    Post-GNN node saliency + edge saliency (saliency-style layer method).
    Returns:
      node_attr_emb: [N, D]   (∂y/∂h_post ⊙ h_post), where h_post is post-GNN, pre-pool node states
      edge_attr_emb: [M(+N), D]  from selected GINE layer (after self-loops are added there)
      node_scores:   [N]      L1 over embedding dims
      edge_scores:   [M(+N)]  L1 over embedding dims
    """
    dev = device or next(model.parameters()).device
    model.eval().to(dev)

    x         = _to_dev(x.long(), dev)
    edge_index= _to_dev(edge_index.long(), dev)
    attr      = _to_dev(attr.long(), dev)
    x_batch   = _to_dev(x_batch.long() if x_batch is not None else None, dev)

    # --- Node saliency at the **post-GNN** tap
    gxa_nodes = LayerGradientXActivation(model, model.node_emb_tap_post)
    node_attr_emb = gxa_nodes.attribute(
        inputs=x,
        target=target,
        additional_forward_args=(edge_index, attr, x_batch),
        attribute_to_layer_input=True
    )  # [N, D]
    node_scores = node_attr_emb.abs().sum(-1)  # [N]

    # --- Edge saliency at chosen GINE layer's edge-embedding tap
    assert 0 <= edge_layer_idx < len(model.gnns), "edge_layer_idx out of range"
    edge_tap = model.gnns[edge_layer_idx].edge_emb_tap
    gxa_edges = LayerGradientXActivation(model, edge_tap)
    edge_attr_emb = gxa_edges.attribute(
        inputs=x,  # same primary input; Captum hooks the specified layer internally
        target=target,
        additional_forward_args=(edge_index, attr, x_batch),
        attribute_to_layer_input=True
    )  # [M+N, D] (that layer appends self-loops)
    edge_scores = edge_attr_emb.abs().sum(-1)  # [M+N]

    return {
        "node_attr_emb": node_attr_emb.detach(),
        "edge_attr_emb": edge_attr_emb.detach(),
        "node_scores": node_scores.detach(),
        "edge_scores": edge_scores.detach()
    }

@torch.no_grad()

def explain_molecular_encoder_gxa(
    model: torch.nn.Module,
    x_dict: dict,
    edge_index_list,
    edge_index_dict: dict,
    chem_idx: int,
    target_class: int = None,
    gine_layer_idx: int = 0,
    device: str = None,
    attribute_to_layer_input: bool = True,
):
    """
    Computes node & edge saliency for the molecule encoder via LayerGradientXActivation.
    Returns a dict with node/edge embeddings and collapsed scores (L1 over features).
    """
    # -------- setup --------
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    if isinstance(device, str):
        device = torch.device(device)

    model = model.to(device).eval()

    # Atom features (can be Long); we do NOT require requires_grad=True because
    # attributions are wrt the chosen layer’s inputs/outputs, not the model input.
    x_atom = x_dict['Atom'].to(device)

    # closure: swap Atom features, run full model (semantic path included), return scalar logit
    def forward_for_one_chem(x_atom_override: torch.Tensor) -> torch.Tensor:
        # x_atom_override has same dtype/shape as x_atom
        local_x = dict(x_dict)
        local_x['Atom'] = x_atom_override
        logits = model(local_x, edge_index_list, edge_index_dict, chemical_label_task=None)  # [N_chem, C]
        # Captum expects a scalar
        return logits[chem_idx, target_class].unsqueeze(0)

    # pick class if not provided
    if target_class is None:
        target_class = pick_target_class(model, x_dict, edge_index_list, edge_index_dict, chem_idx)

    # ---- define taps ----
    # Node tap: per-ATOM embeddings after molecule GNN (pre-pool)
    node_tap = model.molecular_encoder.node_emb_tap_post
    # Edge tap: per-edge embeddings inside a selected GINE layer
    edge_tap = model.molecular_encoder.gnns[gine_layer_idx].edge_emb_tap

    # ---- Captum: LayerGradientXActivation on node & edge taps ----
    gxa_nodes = LayerGradientXActivation(forward_for_one_chem, node_tap)
    gxa_edges = LayerGradientXActivation(forward_for_one_chem, edge_tap)

    # These return tensors with same shape as the layer's (input if attribute_to_layer_input=True, else output).
    node_attr_emb = gxa_nodes.attribute(
        inputs=x_atom,
        attribute_to_layer_input=attribute_to_layer_input
    )
    edge_attr_emb = gxa_edges.attribute(
        inputs=x_atom,
        attribute_to_layer_input=attribute_to_layer_input
    )

    # Collapse feature dimension -> scalar scores
    # node_attr_emb: [N_atoms, D] -> [N_atoms]
    # edge_attr_emb: [M_edges(+N_selfloops), D] -> [M_edges(+N_selfloops)]
    node_scores = node_attr_emb.abs().sum(dim=-1).detach().cpu()
    edge_scores = edge_attr_emb.abs().sum(dim=-1).detach().cpu()

    out = {
        'chem_idx': int(chem_idx),
        'target_class': int(target_class),
        'node_attr_emb': node_attr_emb.detach().cpu(),   # tensor [N_atoms, D]
        'edge_attr_emb': edge_attr_emb.detach().cpu(),   # tensor [M(+selfloops), D]
        'node_scores': node_scores,                      # tensor [N_atoms]
        'edge_scores': edge_scores,                      # tensor [M(+selfloops)]
        # optional hints that help align back to your graph:
        'gine_layer_idx': int(gine_layer_idx),
        'attribute_to_layer_input': bool(attribute_to_layer_input),
    }
    return out

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
        
        
    def forward(self, x_dict, edge_index_list, edge_index_dict, chemical_label_task):
        
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
        
        # this need to be deleted for explainability
        #return  x_dict['Chemical']
    
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

    explainability = True

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

    # tox21
    # with open("./data/tox21/list_tasks_assay_ML.pkl", "rb") as fp:
    #     list_tasks = pickle.load(fp)

    # liver
    list_tasks = ['Liver_2vs']
    
    with open("./data/Liver/best_hyperparams_MolCLR_5_runs.pkl", "rb") as fp:   
            best_hyperparams_MolCLR_5_runs = pickle.load(fp)

    # Semantic
    if args.semantic == "gnn_encoder":

        load_data_results = load_data_graph_dict("./data/Liver/alone/Hetero_subgraph_comptoxAIs_hypernode.pt", args.CL == 1, 
                                "./data/Liver/alone/node_name_numerical_idx_gene.pkl",
                                "./data/Liver/alone/node_name_numerical_idx_chemicals.pkl",
                                "./data/Liver/alone/node_name_numerical_idx_assay.pkl",
                                 "./data/Liver/alone/node_name_numerical_idx_chemlist.pkl",
                                ['Liver_2vs'])

        Hetero_subgraph_comptoxAI = load_data_results[0]
        node_name_numerical_idx_chemicals = load_data_results[1]
        node_name_numerical_idx_gene = load_data_results[2]
        node_name_numerical_idx_assay = load_data_results[3]
        node_name_numerical_idx_chemlist = load_data_results[4]


        if args.CL == 0: 
            with open("./data/Liver/alone/best_hyperparams_semantic_5_runs.pkl", "rb") as fp:   
                best_hyperparams_semantic_5_runs = pickle.load(fp)
        else:
            with open("./data/Liver/alone/best_hyperparams_semantic_CL_5_runs.pkl", "rb") as fp:   
                best_hyperparams_semantic_5_runs = pickle.load(fp)

        warnings.filterwarnings("ignore")

    for init_seed in [127]:

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

            if explainability:
                datadf = pd.read_excel("./data/Liver/datasets_valid_and_splits/" + task_name + "_df.xlsx")
            else:
                datadf = pd.read_excel("../tox21_chemicals_to_visualize.xlsx")
                
            config['init_lr'] = best_hyperparams_MolCLR_5_runs['init_lr']
            config['init_base_lr'] = best_hyperparams_MolCLR_5_runs['init_base_lr']
            config['pred_n_layer'] = best_hyperparams_MolCLR_5_runs['pred_n_layer']
            config['weight_decay'] = best_hyperparams_MolCLR_5_runs['weight_decay']
            config['drop_ratio'] = best_hyperparams_MolCLR_5_runs['drop_ratio']
            config['feat_dim'] = best_hyperparams_MolCLR_5_runs['feat_dim']


            datadf_pos = datadf[datadf['target'] == 1]

            if explainability:
                attribution_df = pd.read_excel("./data/Liver/Liver_.xlsx")
                attribution_df['name'] = attribution_df['Drug Name']
                attribution_df['smiles'] = attribution_df['n.sMILES']


                new_df = datadf_pos.merge(attribution_df[['name','smiles','attribution']])
                all_df = datadf.merge(attribution_df[['name','smiles','attribution']])

                molecule_dataset_positive = MolTestDataset(datadf=new_df, target=1, task = 'classification', task_name = task_name)
                all_dataset = MolTestDataset(datadf=all_df, target=1, task = 'classification', task_name = task_name)

            else:
                datadf['attribution'] = [str(np.zeros(5,dtype=float)) for x in range(datadf.shape[0])]
                all_dataset = MolTestDataset(datadf=datadf, target=1, task = 'classification', task_name = task_name)

            if args.semantic == "":

                model = GINet(config['pred_n_layer'], config['drop_ratio'],config['feat_dim'],args.semantic,**config["model"])
                    
                model_path = "./ckpt/comptoxai/GIN_node_masking/Liver/best_run_models_" + str(init_seed) + "/model_test_" + task_name + ".pth"
                state_dict = torch.load(model_path, map_location=device)
                
                model.load_my_state_dict(state_dict)

                model.eval().to('cpu')


            else:

                model_pretrained = GINet(config['pred_n_layer'], config['drop_ratio'],config['feat_dim'],args.semantic,**config["model"])

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
                subgraph_Hetero_subgraph_comptoxAI.to('cuda:0')

                model = fancy_GNN(subgraph_Hetero_subgraph_comptoxAI.metadata(),args.CL == 1,
                              hidden_channels=hyperparams_semantic['hidden_channels'],
                              out_channels = 2,
                              num_layers = hyperparams_semantic['num_layers'],
                              drop_ratio= hyperparams_semantic['drop_ratio'],
                              norm_type=hyperparams_semantic['norm'],
                              aggregation_type=hyperparams_semantic['aggregation_node'],
                              molecular_embedder= model_pretrained).to(device)


                if args.CL == 1:
                    semmol = torch.load('./ckpt/comptoxai/GIN_node_masking/Liver/best_run_semantics_graphs_' + str(init_seed) + "/alone/semantic_graph_model_CL_test_" + task_name + ".pth")
                else:
                    semmol = torch.load('./ckpt/comptoxai/GIN_node_masking/Liver/best_run_semantics_graphs_' + str(init_seed) + "/alone/semantic_graph_model_test_" + task_name + ".pth")
                

                model.load_state_dict(semmol)
                model.eval().to('cuda:0')


            # methods = [ "Layer_Saliency",
            #     CaptumExplainer(attribution_method="Saliency"),   
            # CaptumExplainer(attribution_method="InputXGradient"),    
            # GNNExplainer(lr=0.01),      
            # GNNExplainer(lr=0.001),
            # PGExplainer(epochs=100, lr=0.001),
            # PGExplainer(epochs=100, lr=0.01)]

            # methods_n = ['Layer_Saliency','Saliency','InputXGradient','GNNExp_01','GNNExp_001','PGExp_01','PGExp_001']
   
            # remember to switch the return in the GINet class
            if explainability:
                for explain_method,explainer_name in zip(['Layer_Saliency'],['Layer_Saliency']):

                    # GNN encoder finetuned
                    if args.semantic == "":

                        if explainer_name == "Layer_Saliency":
                            for molecule_data in tqdm(molecule_dataset_positive):
                                res = layer_gxa_node_and_edge_post(
                                            model,  molecule_data.x, molecule_data.edge_index, molecule_data.edge_attr, molecule_data.batch,
                                            target=molecule_data.y.flatten(), edge_layer_idx=4, device='cpu'
                                        )

                                ei_used = model.gnns[4].last_edge_index  # [2, E_used]
                                non_self = (ei_used[0] != ei_used[1])

                                edge_index_no_self   = ei_used[:, non_self]
                                edge_scores_no_self  = res['edge_scores'][non_self]

                                new_res = {'node_scores':res['node_scores'], 'edge_scores':edge_scores_no_self, 'bond_idx':edge_index_no_self}


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
                                chemical_idx = node_name_numerical_idx_chemicals[molecule_data['name']]

                                res = explain_molecular_encoder_gxa(
                                            model=model,
                                            x_dict=subgraph_Hetero_subgraph_comptoxAI.x_dict,
                                            edge_index_list=subgraph_Hetero_subgraph_comptoxAI.edge_items(),
                                            edge_index_dict=subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                                            chem_idx=chemical_idx,
                                            target_class=1,
                                            gine_layer_idx=4,
                                            device='cuda:0',
                                            attribute_to_layer_input=True,   # or False to attribute to the layer output
                                        )


                                A_C = subgraph_Hetero_subgraph_comptoxAI.edge_index_dict[('Atom','hyper','Chemical')]  # [2, num_hyper]
                                atom_ids = A_C[0, A_C[1] == chemical_idx].unique()
                                node_scores_mol = res['node_scores'][atom_ids.cpu()]

                                gine = model.molecular_encoder.gnns[4]
                                ei_used = gine.last_edge_index                      # [2, E_used]  (set by the forward during G×A)
                                u, v = ei_used

                                atom_mask = torch.zeros(res['node_scores'].numel(), dtype=torch.bool, device=ei_used.device)
                                atom_mask[atom_ids.to(ei_used.device)] = True
                                keep_used = atom_mask[u] & atom_mask[v] 

                                edge_scores_mol_directed = res['edge_scores'][keep_used.cpu()]
                                bond_ei_mol_directed = ei_used[:, keep_used]

                                non_self = (bond_ei_mol_directed[0] != bond_ei_mol_directed[1])
                                bond_ei_mol_directed = bond_ei_mol_directed[:, non_self]
                                edge_scores_mol_directed = edge_scores_mol_directed[non_self.cpu()]

                                undirected_ei, undirected_scores = undirected_edge_scores(
                                            bond_ei_mol_directed, edge_scores_mol_directed.to('cuda:0'), num_nodes=res['node_scores'].numel()
                                        )

                                new_res = {'node_scores':node_scores_mol, 'atom_idx_in_hetero':atom_ids,'edge_scores':edge_scores_mol_directed, 'bond_idx_in_hetero':bond_ei_mol_directed}
                                
                                if args.CL == 0:

                                    with open("./results_Liver/explain_sem/" + explainer_name + "/" + str(init_seed) + "/" + task_name + "/" + molecule_data.name + ".pkl", "wb") as f:
                                        pickle.dump(new_res, f)

                                else:

                                    with open("./results_Liver/explain_sem_CL/" + explainer_name + "/" + str(init_seed) + "/" + task_name + "/" + molecule_data.name + ".pkl", "wb") as f:
                                        pickle.dump(new_res, f)
            

            else:
                # remember to switch the return in the GINet class 


                ### MolCLR pretrained
                molclr_pretrained = GINet(config['pred_n_layer'], config['drop_ratio'],config['feat_dim'],args.semantic,**config["model"])
                    
                model_path = "./ckpt/comptoxai/GIN_node_masking/model.pth"
                state_dict = torch.load(model_path, map_location=device)
                
                molclr_pretrained.load_my_state_dict(state_dict)

                molclr_pretrained.eval().to('cuda:0')

                molclr_pretrained.eval()
                valid_dataset_embeddings_molclr = []
                valid_dataset_embeddings_SemMol = []

                if not(os.path.exists("./results_tox21/explain/_df_embeddings_MolCLR.xlsx")):
                    for molecule_data in tqdm(all_dataset):
                        #print(model(x = molecule_data.x, edge_index = molecule_data.edge_index, attr=molecule_data.edge_attr, x_batch=molecule_data.batch)[-1,:].detach().cpu().reshape(1,-1).shape)
                        valid_dataset_embeddings_molclr.append(molclr_pretrained(x = molecule_data.x.to('cuda:0'), edge_index = molecule_data.edge_index.to('cuda:0'), attr=molecule_data.edge_attr.to('cuda:0'), x_batch=molecule_data.batch.to('cuda:0')).detach().cpu())
                    tensor_validation_embeddings_h = torch.cat(valid_dataset_embeddings_molclr)
                    tsne_embedding_MolCLR = TSNE(n_components=2, learning_rate='auto', init='random', random_state=127).fit_transform(tensor_validation_embeddings_h)
                    df_plot_MolCLR = pd.concat([pd.DataFrame({"0":tsne_embedding_MolCLR[:,0], "1":tsne_embedding_MolCLR[:,1],}),datadf])
                    df_plot_MolCLR.to_xlsx("./results_tox21/explain/_df_embeddings_MolCLR.xlsx",index=False)

                model.eval().to('cuda:0')
                valid_dataset_embeddings_SemMol = model(subgraph_Hetero_subgraph_comptoxAI.x_dict,   edge_index_list=subgraph_Hetero_subgraph_comptoxAI.edge_items(),  edge_index_dict=subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,chemical_label_task="").detach().cpu()
                tsne_embedding_SemMol = TSNE(n_components=2, learning_rate='auto', init='random', random_state=127).fit_transform(valid_dataset_embeddings_SemMol)

                df_plot_SemMol = pd.concat([pd.DataFrame({"0":tsne_embedding_SemMol[:,0], "1":tsne_embedding_SemMol[:,1],}),datadf])
                df_plot_SemMol.to_xlsx("./results_tox21/explain/" + task_name + "_df_embeddings_SemMol.xlsx",index=False)
