import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from sklearn.utils.class_weight import compute_class_weight

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondStereo 
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
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


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(datadf, target, task):
    smiles_data, labels, maccs_data = [], [], []
    smiles_transformed_as_a_graph = []
    for index, row in datadf.iterrows():

        smiles = row['smiles']
        label = row['target']
        maccs = row['maccs']
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and label != '':
            smiles_data.append(smiles)
            smiles_transformed_as_a_graph.append(index)
            maccs_data.append(maccs)
            if task == 'classification':
                labels.append(int(label))
            elif task == 'regression':
                labels.append(float(label))
            else:
                ValueError('task must be either regression or classification')

    return smiles_data, labels, maccs_data, smiles_transformed_as_a_graph



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



def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MolTestDataset(Dataset):
    def __init__(self, datadf, target, task, task_name):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels, self.maccs_data, smiles_transformed_as_a_graph = read_smiles(datadf, target, task)
        self.task = task

        # filter the dataframe according to the smiles to transformed to graph by rdkit
        self.filtered_datadf = datadf.filter(items=smiles_transformed_as_a_graph, axis=0).reset_index(drop=True)

        self.conversion = 1

    def __getitem__(self, index):
        mol_graph_pyg = from_smiles_custom(self.smiles_data[index])
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        maccs_string = self.maccs_data[index]

        #mol = Chem.AddHs(mol)


        x = mol_graph_pyg.x

        edge_index = mol_graph_pyg.edge_index
        edge_attr = mol_graph_pyg.edge_attr
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index] * self.conversion, dtype=torch.float).view(1,-1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self, 
        batch_size, seed, datadf,task_name,
        num_workers, valid_size, test_size, 
        target, task, splitting
    ):
        super(object, self).__init__()
        self.datadf = datadf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.seed = seed
        self.task_name = task_name
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        train_dataset = MolTestDataset(datadf=self.datadf, target=self.target, task=self.task, task_name = self.task_name)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            #num_train = len(train_dataset)
            #indices = list(range(num_train))

            # make a stratified sampling
            #idx_train_val, test_idx = train_test_split(indices, test_size=self.test_size, random_state=self.seed ,stratify=train_dataset.filtered_datadf['target'],shuffle=True)
            #df_train_val = train_dataset.filtered_datadf.filter(items=idx_train_val, axis=0)
            #train_idx, valid_idx = train_test_split(idx_train_val, test_size=self.valid_size, random_state=self.seed ,stratify=df_train_val['target'],shuffle=True)
            #np.random.shuffle(indices)

            train_idx = np.load("../../data/datasets_valid_and_splits/" + self.task_name + "_train_idx.npy")
            valid_idx = np.load("../../data/datasets_valid_and_splits/" + self.task_name + "_valid_idx.npy")
            test_idx = np.load("../../data/datasets_valid_and_splits/" + self.task_name + "_test_idx.npy")


            # compute weight for class imbalance -> only on the training set
            training_df = train_dataset.filtered_datadf.filter(items=train_idx, axis=0).reset_index(drop=True)
            self.class_weight = compute_class_weight('balanced',classes=np.unique(training_df['target']), y = training_df['target'])

            #split = int(np.floor(self.valid_size * num_train))
            #split2 = int(np.floor(self.test_size * num_train))
            #valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False,
            worker_init_fn=seed_worker, generator=g
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False,
            worker_init_fn=seed_worker, generator=g
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False,
            worker_init_fn=seed_worker, generator=g
        )

        return train_loader, valid_loader, test_loader
