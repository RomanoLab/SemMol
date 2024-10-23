import math
import random
import numpy as np
from copy import deepcopy
import pandas as pd

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondDir
from rdkit.Chem.rdchem import BondStereo 

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')


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
FORMAL_CHARGE_LIST = list(range(-5,10))
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



def read_smiles(data_path):
    smiles_data_list = []

    csv_reader = pd.read_csv(data_path)
    for i, row in csv_reader.iterrows():

        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol != None:
            smiles_data_list.append(row['SMILES'])

    return smiles_data_list

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    




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




class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)


    def __getitem__(self, index):
        mol_graph_pyg = from_smiles_custom(self.smiles_data[index])
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()
        M_f = mol_graph_pyg.edge_attr.shape[1]

        # random mask a subgraph of the molecule
        num_mask_nodes = max([1, math.floor(0.25*N)])
        num_mask_edges = max([0, math.floor(0.25*M)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
        mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
        mask_edges_j = [2*i for i in mask_edges_j_single] + [2*i+1 for i in mask_edges_j_single]

        x_i = deepcopy(mol_graph_pyg.x)
        for atom_idx in mask_nodes_i:
            x_i[atom_idx,:] = torch.tensor([0, 0, 11, 9, 9, 5, 0, 2, 2])
        edge_index_i = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_i = torch.zeros((2*(M-num_mask_edges), M_f), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_i:
                edge_index_i[:,count] = mol_graph_pyg.edge_index[:,bond_idx]
                edge_attr_i[count,:] = mol_graph_pyg.edge_attr[bond_idx,:]
                count += 1
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

        x_j = deepcopy(mol_graph_pyg.x)
        for atom_idx in mask_nodes_j:
            x_j[atom_idx,:] = torch.tensor([0, 0, 11, 9, 9, 5, 0, 2, 2])
        edge_index_j = torch.zeros((2, 2*(M-num_mask_edges)), dtype=torch.long)
        edge_attr_j = torch.zeros((2*(M-num_mask_edges), M_f), dtype=torch.long)
        count = 0
        for bond_idx in range(2*M):
            if bond_idx not in mask_edges_j:
                edge_index_j[:,count] = mol_graph_pyg.edge_index[:,bond_idx]
                edge_attr_j[count,:] = mol_graph_pyg.edge_attr[bond_idx,:]
                count += 1
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        
        return data_i, data_j


    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, seed, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.seed = seed

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)

        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        np.save("../../data/valid_idx.npy",valid_idx,allow_pickle=True)


        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, 
                                  worker_init_fn=seed_worker, generator=g)


        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True, 
                                  worker_init_fn=seed_worker,  generator=g)

        return train_loader, valid_loader
