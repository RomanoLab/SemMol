import torch
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from torch_geometric.data import Batch
from torch_geometric.data import Data
from chemutils import get_mol, get_clique_mol

def read_smiles(datadf):
    smiles_data, labels = [], []
    smiles_transformed_as_a_graph = []
    for index, row in datadf.iterrows():

        smiles = row['smiles']
        label = row['target']
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and label != '':
            smiles_data.append(smiles)
            smiles_transformed_as_a_graph.append(index)
            labels.append(int(label))
    return smiles_data, labels, smiles_transformed_as_a_graph


class MoleculeDataset(Dataset):

    def __init__(self, data_file):
        self.smiles_data, self.labels, smiles_transformed_as_a_graph = read_smiles(data_file)

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        smiles = self.smiles_data[idx]
        label = self.labels[idx]
        mol_graph = MolGraph(smiles,label)  
        return mol_graph


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


class MolGraph(object):

    def __init__(self, smiles, label):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.molecule_data = from_smiles_custom(smiles)
        self.x_nosuper = self.molecule_data.x
        self.edge_index_nosuper = self.molecule_data.edge_index
        self.edge_attr_nosuper = self.molecule_data.edge_attr


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




