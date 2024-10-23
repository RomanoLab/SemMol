from tqdm import tqdm
import os
import pickle
import random
import numpy as np
import pandas as pd
import sys
sys.path.append('../MolCLR/dataset_test')
from dataset_test import MolTestDataset

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondStereo                                                                                                                                        
RDLogger.DisableLog('rdApp.*')  

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.explain import GNNExplainer
from torch_geometric.explain import Explainer
import torch_geometric

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

# simplified version of the model
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        
    def forward(self, x, edge_index):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        return self.propagate(edge_index, x=x)

    
    def message(self, x_j):
        return x_j

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
    def __init__(self, pred_n_layer,drop_ratio,feat_dim,
        task='classification', num_layer=5, emb_dim=300, 
        pool='mean', pred_act='softplus'
    ):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = 'classification'

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

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
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


    def forward(self, x, edge_index):
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5]) + self.x_embedding7(x[:,6]) + self.x_embedding8(x[:,7]) + self.x_embedding9(x[:,8])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, batch=None)

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

def get_explanations(list_task_tox21):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read the configuration of the best hyperparams for each model part
    with open("./data/best_hyperparams_MolCLR_5_runs.pkl", "rb") as fp:   
        best_hyperparams_MolCLR_5_runs = pickle.load(fp)

    for init_seed in [127,128,129,130,131]:

        torch.manual_seed(init_seed)
        random.seed(init_seed)
        np.random.seed(init_seed)

        for assay in list_task_tox21:
            hyperparams_MolCLR_finetuning = eval(best_hyperparams_MolCLR_5_runs[assay])
            best_results_graph_hyperparams = hyperparams_MolCLR_finetuning

            ## run the finetune to obtain the trained model
            config = yaml.load(open("./MolCLR/config_finetune.yaml", "r"), Loader=yaml.FullLoader)
            config['seed'] = init_seed
            config['dataset']['task'] = 'classification'
            config['dataset']['target'] = 1

            datadf = pd.read_excel("./data/datasets_valid_and_splits/" + assay + "_df.xlsx")

            torch.manual_seed(init_seed)
            random.seed(init_seed)
            np.random.seed(init_seed)

            config['init_lr'] = best_results_graph_hyperparams['init_lr']
            config['init_base_lr'] = best_results_graph_hyperparams['init_base_lr']
            config['pred_n_layer'] = best_results_graph_hyperparams['pred_n_layer']
            config['weight_decay'] = best_results_graph_hyperparams['weight_decay']
            config['drop_ratio'] = best_results_graph_hyperparams['drop_ratio']
            config['feat_dim'] = best_results_graph_hyperparams['feat_dim']
            
            model = GINet(config['pred_n_layer'], config['drop_ratio'],config['feat_dim'],**config["model"]).to(device)
            
            model_path = "./MolCLR/ckpt/best_run_models_" + str(init_seed) + "/model_test_" + assay + ".pth"
            state_dict = torch.load(model_path, map_location=device)
        
            model.load_my_state_dict(state_dict)
            

            datadf_pos = datadf[datadf['target'] == 1]


            train_dataset = MolTestDataset(datadf=datadf_pos, target=1, task = "classification", task_name = assay)

            explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(),
            explanation_type='phenomenon',
            node_mask_type=None,
            edge_mask_type='object',

            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw',
            ),
            )

            if not os.path.exists("./MolCLR/results/gnn_explainer_MolCLR_pos_" + str(init_seed) + "/" + assay):
                os.makedirs("./results/gnn_xai_" + str(init_seed) + "/" + assay)
            else:
                continue

            for compound_graph,compund_name in tqdm(zip(train_dataset,datadf_pos['name'])):

                compund_name = compund_name.replace("'","")
                compund_name = compund_name.replace("/","")
                compund_name = compund_name.replace(" ","")
                compund_name = compund_name.replace('"','')
                compund_name = compund_name.replace(':','')

                explanation_compound = explainer(x = compound_graph[0].x.to(device), edge_index = compound_graph[0].edge_index.to(device), target = compound_graph[0].y.flatten().to(device))

                explanation_compound['name'] = compund_name

                try:
                    torch.save(explanation_compound,"./results/gnn_explainer_xai_" + str(init_seed) + "/" + assay + "/" + compund_name + ".pth")
                except:
                    continue

    return 1

def main():

    with open("./data/list_tasks.pkl", "rb") as fp:
        list_task_tox21 = pickle.load(fp)

    get_explanations(list_task_tox21)

if __name__ == "__main__":
    main()