import pandas as pd
from comptox_ai.db.graph_db import GraphDB
from tqdm import tqdm
import pickle
import numpy as np
import yaml
import sys

import torch
from torch_geometric.data import HeteroData

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondStereo                                                                                                                                        
RDLogger.DisableLog('rdApp.*')  

sys.path.append('../MolCLR/dataset_test')
from dataset_test import MolTestDataset
sys.path.append('../MolCLR/models/ginet_molclr')
from ginet_molclr import GINet

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
    
def create_base_KG(db, list_task_tox21):

    ### CHEMICALS nodes
    all_chemicals_involved = []
    node_name_numerical_idx_chemicals = {}
    counter_idx_chemicals = 0

    for assay in list_task_tox21:
        #retrieve all the chemicals involved in the predictiont task
        dataset_prediction_assay = pd.read_excel("./datasets_valid_and_splits/" + assay + "_df.xlsx")
        
        chemicals_name = list(dataset_prediction_assay['name'].values)
        
        all_chemicals_involved += chemicals_name
        
    unique_chemicals = list(set(all_chemicals_involved))
        
    for chem in unique_chemicals:
        node_name_numerical_idx_chemicals[chem] = counter_idx_chemicals
        counter_idx_chemicals+=1

    ### CHEMICAL-ASSAY relations

    with open("./unique_dict_chemical_of_interest_1.pkl", "rb") as fp:   
        unique_dict_chemical_of_interest_1 = pickle.load( fp)

    with open("./unique_dict_chemical_of_interest_0.pkl", "rb") as fp:   
        unique_dict_chemical_of_interest_0 = pickle.load( fp)


    ### CHEMICAL - GENE relations
    query_chemical_genes = db.run_cypher("MATCH (n:Chemical)-[r]-(m:Gene) RETURN n,r,m")

    # filter the relationship between chemicals and genes according to the chemicals available

    triplets_to_keep_chemical_increase_gene = []
    triplets_to_keep_chemical_decrease_gene = []
    triplets_to_keep_chemical_binds_gene = []

    for triplets in tqdm(query_chemical_genes):
        if 'commonName' in triplets['r'][0].keys():
            chemical = triplets['r'][0]['commonName']
            if chemical in all_chemicals_involved:
                if triplets['r'][1] == "CHEMICALINCREASESEXPRESSION":
                    triplets_to_keep_chemical_increase_gene.append(triplets['r'])
                elif triplets['r'][1] == "CHEMICALDECREASESEXPRESSION":
                    triplets_to_keep_chemical_decrease_gene.append(triplets['r'])
                elif triplets['r'][1] == "CHEMICALBINDSGENE":
                    triplets_to_keep_chemical_binds_gene.append(triplets['r'])
        
    ### GENES nodes

    # extract genes from these triplets
    unique_gene = []
    unique_gene_symbol = []

    for triple in triplets_to_keep_chemical_increase_gene:
        unique_gene.append(triple[2]['commonName'])
        unique_gene_symbol.append(triple[2]['geneSymbol'])
        
    for triple in triplets_to_keep_chemical_decrease_gene:
        unique_gene.append(triple[2]['commonName'])
        unique_gene_symbol.append(triple[2]['geneSymbol'])
        
    for triple in triplets_to_keep_chemical_binds_gene:
        unique_gene.append(triple[2]['commonName'])
        unique_gene_symbol.append(triple[2]['geneSymbol'])

    unique_gene_symbol = list(set(unique_gene_symbol))

    node_name_numerical_idx_gene = {}
    counter_idx_gene = 0

    for gene_symbol in unique_gene_symbol:
        node_name_numerical_idx_gene[gene_symbol] = counter_idx_gene
        counter_idx_gene+=1

    ### ASSAY - GENE relation -> from literature and tox21 protocols
    dict_assay_gene_involved = {'tox21-ahr-p1':["AHR"],
                            'tox21-ap1-agonist-p1':["FOS","JUN","FOSL2","FOSL1","JUNB","FOSB","JUND"],
                            'tox21-ar-bla-agonist-p1':["AR"],
                            'tox21-ar-bla-antagonist-p1':["AR"],
                            'tox21-ar-mda-kb2-luc-agonist-p1':["AR"],
                            'tox21-ar-mda-kb2-luc-agonist-p3':["AR"],
                            'tox21-ar-mda-kb2-luc-antagonist-p1':["AR"],
                            'tox21-ar-mda-kb2-luc-antagonist-p2':["AR"],
                            'tox21-are-bla-p1':["NFE2"],
                            'tox21-aromatase-p1':["CYP19A1"],
                            'tox21-car-agonist-p1':["NR1I3"],
                            'tox21-car-antagonist-p1':["NR1I3"],
                            'tox21-casp3-cho-p1':['CASP3'],
                            'tox21-casp3-hepg2-p1':['CASP3'],
                            'tox21-elg1-luc-agonist-p1':['ATAD5'],
                            'tox21-er-bla-agonist-p2':['ESR1'],
                            'tox21-er-bla-antagonist-p1':['ESR1'],
                            'tox21-er-luc-bg1-4e2-agonist-p2':['ESR1'],
                            'tox21-er-luc-bg1-4e2-agonist-p4':['ESR1'],
                            'tox21-er-luc-bg1-4e2-antagonist-p1':['ESR1'],
                            'tox21-er-luc-bg1-4e2-antagonist-p2':['ESR1'],
                            'tox21-erb-bla-antagonist-p1':["ESR2"],
                            'tox21-erb-bla-p1':["ESR2"],
                            'tox21-esre-bla-p1':["ATF6"],
                            'tox21-fxr-bla-agonist-p2':["NR1H4"],
                            'tox21-fxr-bla-antagonist-p1':["NR1H4"],
                            'tox21-gh3-tre-antagonist-p1':['THRB'],
                            'tox21-gr-hela-bla-agonist-p1':['NR3C1'],
                            'tox21-gr-hela-bla-antagonist-p1':['NR3C1'],
                            'tox21-h2ax-cho-p2':["H2AX"],
                            'tox21-hdac-p1':['HDAC1','HDAC2'],
                            'tox21-hre-bla-agonist-p1':['HIF1A'],
                            'tox21-hse-bla-p1':['HSPA5','HSPA8','HSPA1L','HSPA6','HSPA1A','HSPA4L','HSPA2','HSPA12A','HSPA13','HSPA1B','HSPA14','HSPA9','HSPA12B','HSPA4','HSPA7'],
                            'tox21-mitotox-p1':[],
                            'tox21-p53-bla-p1':['TP53'],
                            'tox21-ppard-bla-agonist-p1':['PPARD'],
                            'tox21-ppard-bla-antagonist-p1':['PPARD'],
                            'tox21-pparg-bla-agonist-p1':['PPARG'],
                            'tox21-pparg-bla-antagonist-p1':['PPARG'],
                            'tox21-pr-bla-agonist-p1':['PGR'],
                            'tox21-pr-bla-antagonist-p1':['PGR'],
                            'tox21-pxr-p1':['NR1I2'],
                            'tox21-rar-agonist-p1':['RARA','RARB','RARG'],
                            'tox21-rar-antagonist-p2':['RARA','RARB','RARG'],
                            'tox21-ror-cho-antagonist-p1':['RORC'],
                            'tox21-rt-viability-hepg2-p1':[],
                            'tox21-rxr-bla-agonist-p1':['RXRA'],
                            'tox21-sbe-bla-antagonist-p1':['SMAD2','SMAD3'],
                            'tox21-shh-3t3-gli3-antagonist-p1':['GLI1'],
                            'tox21-tshr-agonist-p1':['TSHR'],
                            'tox21-tshr-antagonist-p1':['TSHR'],
                            'tox21-vdr-bla-antagonist-p1':['VDR']}

    ### ASSAY nodes
    node_name_numerical_idx_assay = {}
    counter_idx_assay = 0

    for key,value in dict_assay_gene_involved.items():
        if key in list_task_tox21:
            node_name_numerical_idx_assay[key] = counter_idx_assay
            counter_idx_assay+=1

    ### GENE - GENE relation
    gengen = db.run_cypher("MATCH (n:Gene)-[r]-(m:Gene) RETURN n,r,m")
    gen_gen_interaction_of_interest = []
    for gen_gen_interaction in gengen:
        if (gen_gen_interaction['r'][0]['geneSymbol'] in node_name_numerical_idx_gene.keys()) and (gen_gen_interaction['r'][2]['geneSymbol'] in node_name_numerical_idx_gene.keys()):
            gen_gen_interaction_of_interest.append(gen_gen_interaction)


        

    ### Pytorch geometric Data object creation
    Hetero_subgraph_comptoxAI = HeteroData()

    # define relations as head - relation - tail
    entitytype_relationname_entitytype_dict = [['Chemical','CHEMICALHASACTIVEASSAY','Assay'],
                                            ['Chemical','CHEMICALHASINACTIVEASSAY','Assay'],
                                            ['Chemical','CHEMICALINCREASESEXPRESSION','Gene'],
                                            ['Chemical','CHEMICALDECREASESEXPRESSION','Gene'],
                                            ['Chemical','CHEMICALBINDSGENE','Gene'],
                                            ['Assay','ASSAYTARGETGENE','Gene'],
                                            ['Gene','GENEINTERACTSWITHGENE','Gene'],
                                            ['ChemicalList','CHEMICALHASACTIVEASSAY','Chemical'],
                                            ]


    for relation_type in entitytype_relationname_entitytype_dict:
        
        head = relation_type[0]
        relation = relation_type[1]
        tail = relation_type[2]
        
        edge_tensors_list = []
        
        if relation=="CHEMICALHASACTIVEASSAY":
            for key,values in tqdm(unique_dict_chemical_of_interest_1.items()):
                if key in list_task_tox21:
                    for chemical in values:
                        edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemicals[chemical],node_name_numerical_idx_assay[key]]], dtype=torch.long))
                
        elif relation=="CHEMICALHASINACTIVEASSAY":
            for key,values in tqdm(unique_dict_chemical_of_interest_0.items()):
                if key in list_task_tox21:
                    for chemical in values:
                        edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemicals[chemical],node_name_numerical_idx_assay[key]]], dtype=torch.long))

        elif relation=="CHEMICALINCREASESEXPRESSION":
            for chem_gen in tqdm(triplets_to_keep_chemical_increase_gene):
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemicals[chem_gen[0]['commonName']],node_name_numerical_idx_gene[chem_gen[2]['geneSymbol']]]], dtype=torch.long))

            
        elif relation=="CHEMICALDECREASESEXPRESSION":
            for chem_gen in tqdm(triplets_to_keep_chemical_decrease_gene):
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemicals[chem_gen[0]['commonName']],node_name_numerical_idx_gene[chem_gen[2]['geneSymbol']]]], dtype=torch.long))

        elif relation=="CHEMICALBINDSGENE":
            for chem_gen in tqdm(triplets_to_keep_chemical_binds_gene):
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemicals[chem_gen[0]['commonName']],node_name_numerical_idx_gene[chem_gen[2]['geneSymbol']]]], dtype=torch.long))

        elif relation=="ASSAYTARGETGENE":
            for key,values in tqdm(dict_assay_gene_involved.items()):
                if key in list_task_tox21:
                    for gene in values:
                        edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_assay[key],node_name_numerical_idx_gene[gene]]], dtype=torch.long))

        elif relation=="GENEINTERACTSWITHGENE":
            for gene_gene_interaction in tqdm(gen_gen_interaction_of_interest):
                gene_gene_interaction = gene_gene_interaction['r']
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_gene[gene_gene_interaction[0]['geneSymbol']],node_name_numerical_idx_gene[gene_gene_interaction[2]['geneSymbol']]]], dtype=torch.long))

                
        edge_tensor = torch.cat(edge_tensors_list,dim=0)

        Hetero_subgraph_comptoxAI[head,relation,tail].edge_index = edge_tensor.t().contiguous()
        

        
    # add number of nodes
    Hetero_subgraph_comptoxAI['Chemical'].num_nodes = len(unique_chemicals)
    Hetero_subgraph_comptoxAI['Assay'].num_nodes = len(list_task_tox21)
    Hetero_subgraph_comptoxAI['Gene'].num_nodes = len(unique_gene_symbol)



    ## add chemical list
    Hetero_subgraph_comptoxAI = torch.load("./MolCLR/data/Hetero_subgraph_comptoxAIs_hypernode.pt")

    with open("./MolCLR/data/node_name_numerical_idx_chemicals.pkl", "rb") as fp:
        node_name_numerical_idx_chemicals = pickle.load(fp)
        
    with open("./MolCLR/data/node_name_numerical_idx_gene.pkl", "rb") as fp:
        previous_node_name_numerical_idx_gene = pickle.load(fp)
        
    chem_chemlist = pd.read_csv("./chem_chemlist.csv")

    with open('./MolCLR/data/node_name_numerical_idx_chemlist.pkl', 'rb') as handle:
        node_name_numerical_idx_chemlist = pickle.load(handle)
        

    edge_tensors_list = []

    for index, row in tqdm(chem_chemlist[chem_chemlist['g.commonName'].isin(node_name_numerical_idx_chemicals.keys())].iterrows()):
        head = row['n.commonName']
        relation = row['type(r)']
        tail = row['g.commonName']
        
            
        edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemlist[head],node_name_numerical_idx_chemicals[tail]]], dtype=torch.long))

    edge_tensor = torch.cat(edge_tensors_list,dim=0)


    Hetero_subgraph_comptoxAI['ChemicalList',relation,'Chemical'].edge_index = edge_tensor.t().contiguous()

    Hetero_subgraph_comptoxAI['ChemicalList'].num_nodes = len(node_name_numerical_idx_chemlist)

    Hetero_subgraph_comptoxAI['ChemicalList'].x = torch.tensor(np.random.uniform(low=0,high=1,size=len(node_name_numerical_idx_chemlist)),dtype=torch.float32).reshape(-1,1)
        


    # save the graph
    torch.save(Hetero_subgraph_comptoxAI,"./base_Hetero_subgraph_comptoxAIs.pt")


    # save the results
    with open('./dict_chemical_list_compounds.json', 'wb') as file:
        json.dump(dict_chemical_list_compounds, file)

    # and the dictionary with the node idx for each nodetype
    with open("./node_name_numerical_idx_gene.pkl", "wb") as fp:
        json.dump(node_name_numerical_idx_gene, fp)
        
    with open("./node_name_numerical_idx_chemicals.pkl", "wb") as fp:
        json.dump(node_name_numerical_idx_chemicals, fp)
        
    with open("./node_name_numerical_idx_assay.pkl", "wb") as fp:
        json.dump(node_name_numerical_idx_assay, fp)



        

    return node_name_numerical_idx_chemicals, unique_gene_symbol

def add_node_feature(list_task_tox21, node_name_numerical_idx_chemicals, unique_gene_symbol):

    Hetero_subgraph_comptoxAI = torch.load("./data/base_Hetero_subgraph_comptoxAIs.pt")

    total_dataset_prediction_assay = pd.DataFrame(columns=['name','maccs','smiles','target'])

    for assay in tqdm(list_task_tox21):
        #retrieve all the chemicals involved in the predictiont task
        dataset_prediction_assay = pd.read_excel("./datasets_valid_and_splits/" + assay + "_df.xlsx")
        total_dataset_prediction_assay = pd.concat([total_dataset_prediction_assay,dataset_prediction_assay],ignore_index=True)

    total_dataset_prediction_assay = total_dataset_prediction_assay[['name','smiles','maccs']].drop_duplicates()
    total_dataset_prediction_assay['target'] = [1 for i in range(len(total_dataset_prediction_assay))]

    # create the overall pretraining dataset
    all_dataset_tox21 = MolTestDataset(datadf=total_dataset_prediction_assay, target=1, task = "classification", task_name = "")
    
    # load the pretrained model
        
    # read config file
    config_pretrain = yaml.load(open("../MolCLR/config_pretrain.yaml", "r"), Loader=yaml.FullLoader)

    # create a model instance
    model_pretrain = GINet(**config_pretrain["model"])

    # load the model pretrained weight
    model_pretrain.load_state_dict(torch.load('../MolCLR/ckpt/model.pth',map_location='cuda:0'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pretrain.eval()
    tox21_dataset_embeddings_h_emb = []
    dict_drug_emb_node_id = {}

    # compute the embedding for each chemical
    model_pretrain.to(device)
    for data,data_name in tqdm(zip(all_dataset_tox21,total_dataset_prediction_assay['name'])):    
        data = data[0]
        data.to(device)
        h_emb,h,out = model_pretrain(data)
        tox21_dataset_embeddings_h_emb.append(h_emb.detach().cpu())
        dict_drug_emb_node_id[node_name_numerical_idx_chemicals[data_name]] = h_emb.detach().cpu()


    dict_drug_emb_node_id = dict(sorted(dict_drug_emb_node_id.items()))
    list_of_embeddings = list(dict_drug_emb_node_id.values())
    tensor_of_embeddings = torch.cat(list_of_embeddings,dim=0)

    # assing the node features
    Hetero_subgraph_comptoxAI['Chemical'].x = tensor_of_embeddings
    Hetero_subgraph_comptoxAI['Gene'].x = torch.tensor(np.random.uniform(low=0,high=1,size=len(unique_gene_symbol)),dtype=torch.float32).reshape(-1,1)
    Hetero_subgraph_comptoxAI['Assay'].x = torch.tensor(np.random.uniform(low=0,high=1,size=len(list_task_tox21)),dtype=torch.float32).reshape(-1,1)

    torch.save(Hetero_subgraph_comptoxAI,"./Hetero_subgraph_comptoxAIs.pt")

    return 1

def create_hierarchical_KG(list_task_tox21, node_name_numerical_idx_chemicals, Hetero_subgraph_comptoxAI):

    Hetero_subgraph_comptoxAI = torch.load("./data/base_Hetero_subgraph_comptoxAIs.pt")

    # obtain the graph structure for each molecule
    total_dataset_prediction_assay = pd.DataFrame(columns=['name','maccs','smiles','target'])

    for assay in tqdm(list_task_tox21):
        #retrieve all the chemicals involved in the predictiont task
        dataset_prediction_assay = pd.read_excel("./datasets_valid_and_splits/" + assay + "_df.xlsx")
        total_dataset_prediction_assay = pd.concat([total_dataset_prediction_assay,dataset_prediction_assay],ignore_index=True)
        

    total_dataset_prediction_assay = total_dataset_prediction_assay[['name','smiles','maccs']].drop_duplicates()
    total_dataset_prediction_assay['target'] = [1 for i in range(len(total_dataset_prediction_assay))]

    all_dataset_tox21 = MolTestDataset(datadf=total_dataset_prediction_assay, target=1, task = "classification", task_name = "")
        
    # create a dict with the graph molecules as values, sorted for node in the semnatic graph

    dict_node_graph = {}

    for data,data_name in tqdm(zip(all_dataset_tox21,total_dataset_prediction_assay['name'])): 
        data = data[0]
        dict_node_graph[node_name_numerical_idx_chemicals[data_name]] = data

    dict_node_graph_sorted = dict(sorted(dict_node_graph.items()))


    # create the hyperedge and the atom node features
    atom_counter = 0
    atom_idx_counter = 0
    node_features = []
    edge_tensors_list_hyperedge = []
    edge_tensors_list_bonds = []
    edge_attribute_list_bonds = []

    # select a node in the grap
    for node_name,node_id in tqdm(node_name_numerical_idx_chemicals.items()):
        
        #retrieve the graph structure
        molecule_graph_i = dict_node_graph_sorted[node_id]
        
        if node_id == 0:
        
            for tens_atom_node in molecule_graph_i.x:

                edge_tensors_list_hyperedge.append(torch.tensor([[atom_counter,node_id]], dtype=torch.long))
                atom_counter+=1
                atom_idx_counter+=1
                node_features.append(tens_atom_node.reshape(1,-1))
            
            for e in range(molecule_graph_i.edge_index[0].shape[0]):
                edge_tensors_list_bonds.append(torch.tensor([[molecule_graph_i.edge_index[0][e],molecule_graph_i.edge_index[1][e]]], dtype=torch.long))
                edge_attribute_list_bonds.append(molecule_graph_i.edge_attr[e].reshape(1,-1))
            

        else:
            
            for tens_atom_node in molecule_graph_i.x:

                edge_tensors_list_hyperedge.append(torch.tensor([[atom_counter,node_id]], dtype=torch.long))
                atom_counter+=1
                node_features.append(tens_atom_node.reshape(1,-1))
            
            for e in range(molecule_graph_i.edge_index[0].shape[0]):
                edge_tensors_list_bonds.append(torch.tensor([[molecule_graph_i.edge_index[0][e] + atom_idx_counter,molecule_graph_i.edge_index[1][e]+atom_idx_counter]], dtype=torch.long))
                edge_attribute_list_bonds.append(molecule_graph_i.edge_attr[e].reshape(1,-1))
                
            atom_idx_counter += molecule_graph_i.x.shape[0]
        

    
    atom_node_features_tensor = torch.cat(node_features,dim=0)
    edge_tensors_hyperedge = torch.cat(edge_tensors_list_hyperedge,dim=0)
    edge_tensors_bonds = torch.cat(edge_tensors_list_bonds,dim=0)
    edge_attribute_bonds = torch.cat(edge_attribute_list_bonds,dim=0)

    ## add to the graph the hyperedge ('Atom','hyper','Chemical')
    Hetero_subgraph_comptoxAI['Atom','hyper','Chemical'].edge_index = edge_tensors_hyperedge.t().contiguous()
    Hetero_subgraph_comptoxAI['Atom','bond','Atom'].edge_index = edge_tensors_bonds.t().contiguous()
    Hetero_subgraph_comptoxAI['Atom','bond','Atom'].edge_attr = edge_attribute_bonds.contiguous()


    # # add number of atoms node
    Hetero_subgraph_comptoxAI['Atom'].num_nodes = atom_counter

    ## add the node features:
    # gene and assay as scalar value
    Hetero_subgraph_comptoxAI['Gene'].x = torch.tensor(np.random.uniform(low=0,high=1,size=20911),dtype=torch.float32).reshape(-1,1)
    Hetero_subgraph_comptoxAI['Assay'].x = torch.tensor(np.random.uniform(low=0,high=1,size=len(list_task_tox21)),dtype=torch.float32).reshape(-1,1)

    # atom the new node features
    Hetero_subgraph_comptoxAI['Atom'].x = torch.tensor(atom_node_features_tensor,dtype=torch.long)

    # chemicals inizialized with 0
    Hetero_subgraph_comptoxAI['Chemical'].x = torch.zeros((Hetero_subgraph_comptoxAI['Chemical'].num_nodes,300),dtype=torch.float32)

    torch.save(Hetero_subgraph_comptoxAI,"./Hetero_subgraph_comptoxAIs_hypernode.pt")

    return 1

def add_dataset_to_graph(dataset, chem_gen, chem_assay, chem_chemlist, gen_gen, assay_ML, dict_assay_gene_involved, assay_gen,type_graph):

    assay_repo = assay_ML[0].split("_")[0]
    
    ######## create the dictionaries

    # chemical nodes
    node_name_numerical_idx_chemicals = {}
    counter_idx_chemicals = 0
    
    unique_chemicals = list(set(dataset['name']))
    
    for chem in unique_chemicals:
        node_name_numerical_idx_chemicals[chem] = counter_idx_chemicals
        counter_idx_chemicals+=1

    # gene nodes
    unique_gene_symbol = list(set(gen_gen['g.geneSymbol']).union(set(gen_gen['m.geneSymbol'])).union(set(chem_gen['g.geneSymbol'])))
    node_name_numerical_idx_gene = {}
    counter_idx_gene = 0

    for gene_symbol in unique_gene_symbol:
        node_name_numerical_idx_gene[gene_symbol] = counter_idx_gene
        counter_idx_gene+=1
    
    # assay nodes
    if (assay_gen.shape[0]!= 0):
        unique_assay = list(set(chem_assay['c.commonName']).union(set(assay_gen['Assay'])))
    else:
        unique_assay = list(set(chem_assay['c.commonName']).union(set(assay_ML)))
            
    node_name_numerical_idx_assay = {}
    counter_idx_assay= 0

    for assay_name in unique_assay:
        node_name_numerical_idx_assay[assay_name] = counter_idx_assay
        counter_idx_assay+=1

    # chemlist
    unique_chemlist = list(set(chem_chemlist['c.commonName']))
    node_name_numerical_idx_chemlist = {}
    counter_idx_chemlist= 0

    for chemlist_name in unique_chemlist:
        node_name_numerical_idx_chemlist[chemlist_name] = counter_idx_chemlist
        counter_idx_chemlist+=1


    ######## make the KG
    Hetero_subgraph_comptoxAI = HeteroData()
    entitytype_relationname_entitytype_dict = [['Chemical','CHEMICALHASACTIVEASSAY','Assay'],
                                               ['Chemical','CHEMICALHASINACTIVEASSAY','Assay'],
                                               ['Chemical','CHEMICALINCREASESEXPRESSION','Gene'],
                                               ['Chemical','CHEMICALDECREASESEXPRESSION','Gene'],
                                               ['Chemical','CHEMICALBINDSGENE','Gene'],
                                               ['Assay','ASSAYTARGETGENE','Gene'],
                                               ['Gene','GENEINTERACTSWITHGENE','Gene'],
                                              ]


    for relation_type in entitytype_relationname_entitytype_dict:
        print("adding relation:")
        print(relation_type)
        h = relation_type[0]
        relation = relation_type[1]
        t = relation_type[2]

        edge_tensors_list = []

        if relation in ["CHEMICALHASACTIVEASSAY","CHEMICALHASINACTIVEASSAY"]:
            for index,row in tqdm(chem_assay[chem_assay['type(r)'] == relation].iterrows()):
                head = dataset[dataset['cid'] == str(row['n.xrefPubchemCID'])]['name'].item()
                tail = row['c.commonName']
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemicals[head],node_name_numerical_idx_assay[tail]]], dtype=torch.long))

        elif relation in ["CHEMICALINCREASESEXPRESSION","CHEMICALDECREASESEXPRESSION",'CHEMICALBINDSGENE']:
            for index,row in tqdm(chem_gen[chem_gen['type(r)'] == relation].iterrows()):
                head = dataset[dataset['cid'] == str(row['n.xrefPubchemCID'])]['name'].item()
                tail = row['g.geneSymbol']
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemicals[head],node_name_numerical_idx_gene[tail]]], dtype=torch.long))

        elif relation=="ASSAYTARGETGENE":
            for key,values in tqdm(dict_assay_gene_involved.items()):
                if key in node_name_numerical_idx_assay.keys():
                    for gene in values:
                        edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_assay[key],node_name_numerical_idx_gene[gene]]], dtype=torch.long))
          
            for index,row in tqdm(assay_gen.iterrows()):
                head = row['Assay']
                relation = row['relation']
                tail = row['Gene']
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_assay[head],node_name_numerical_idx_gene[tail]]], dtype=torch.long))
                        
        elif relation=="GENEINTERACTSWITHGENE":
            for index,row in tqdm(gen_gen.iterrows()):
                head = row['g.geneSymbol']
                tail = row['m.geneSymbol']
                edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_gene[head],node_name_numerical_idx_gene[tail]]], dtype=torch.long))

        edge_tensor = torch.cat(edge_tensors_list,dim=0)

        Hetero_subgraph_comptoxAI[h,relation,t].edge_index = edge_tensor.t().contiguous()

    # add number of nodes
    Hetero_subgraph_comptoxAI['Chemical'].num_nodes = len(node_name_numerical_idx_chemicals)
    Hetero_subgraph_comptoxAI['Assay'].num_nodes = len(node_name_numerical_idx_assay)
    Hetero_subgraph_comptoxAI['Gene'].num_nodes = len(node_name_numerical_idx_gene)

    Hetero_subgraph_comptoxAI['Gene'].x = torch.tensor(np.random.uniform(low=0,high=1,size=len(node_name_numerical_idx_gene)),dtype=torch.float32).reshape(-1,1)
    Hetero_subgraph_comptoxAI['Assay'].x = torch.tensor(np.random.uniform(low=0,high=1,size=len(node_name_numerical_idx_assay)),dtype=torch.float32).reshape(-1,1)

    with open('./MolCLR/data/' + assay_repo + '/' + type_graph + '/node_name_numerical_idx_chemicals.pkl', 'wb') as handle:
        pickle.dump(node_name_numerical_idx_chemicals, handle)

    with open('./MolCLR/data/' + assay_repo + '/' + type_graph + '/node_name_numerical_idx_assay.pkl', 'wb') as handle:
        pickle.dump(node_name_numerical_idx_assay, handle)

    with open('./MolCLR/data/' + assay_repo + '/' + type_graph + '/node_name_numerical_idx_gene.pkl', 'wb') as handle:
        pickle.dump(node_name_numerical_idx_gene, handle)

    with open('./MolCLR/data/' + assay_repo + '/' + type_graph + '/node_name_numerical_idx_chemlist.pkl', 'wb') as handle:
        pickle.dump(node_name_numerical_idx_chemlist, handle)


    ######## chemical embeddings

    # chemical as 
    dataset_chem = MolTestDataset(datadf=dataset[['name','smiles','maccs','target']], target=1, task = "classification", task_name = "")

    #### load the pretrained model

    # read config file
    config_pretrain = yaml.load(open("./MolCLR/config.yaml", "r"), Loader=yaml.FullLoader)

    # create a model instance
    model_pretrain = GINet_pretrained(**config_pretrain["model"])


    # load the model pretrained weight
    model_pretrain.load_state_dict(torch.load('./MolCLR/ckpt/comptoxai/GIN_node_masking/model.pth',map_location='cuda:0'))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # obtain the validation samples' embedding

    model_pretrain.eval()
    new_embed_to_add = []
    dict_drug_emb_node_id = {}

    model_pretrain.to(device)
    for data,data_name in tqdm(zip(dataset_chem,dataset['name'])):    
        data = data[0]
        data.to(device)
        h_emb,h,out = model_pretrain(data)
        new_embed_to_add.append(h_emb.detach().cpu())
        dict_drug_emb_node_id[node_name_numerical_idx_chemicals[data_name]] = h_emb.detach().cpu()


    dict_drug_emb_node_id = dict(sorted(dict_drug_emb_node_id.items()))
    list_of_embeddings = list(dict_drug_emb_node_id.values())
    tensor_of_embeddings = torch.cat(list_of_embeddings,dim=0)

    Hetero_subgraph_comptoxAI['Chemical'].x = tensor_of_embeddings

    torch.save(Hetero_subgraph_comptoxAI,"./MolCLR/data/" + assay_repo + "/" + type_graph + "/Hetero_subgraph_comptoxAIs.pt")
    
    print(Hetero_subgraph_comptoxAI)
    
    ####### make the hypernode graph
    
    ## create a dict with the graph molecules as values, sorted for node in the semnatic graph
    dict_node_graph = {}

    for data,data_name in tqdm(zip(dataset_chem,dataset['name'])):
        data = data[0]
        dict_node_graph[node_name_numerical_idx_chemicals[data_name]] = data

    dict_node_graph_sorted = dict(sorted(dict_node_graph.items()))


    ######### create the hyperedge and the atom node features

    atom_counter = 0
    atom_idx_counter = 0
    node_features = []
    edge_tensors_list_hyperedge = []
    edge_tensors_list_bonds = []
    edge_attribute_list_bonds = []

    # select a node in the grap
    for node_name,node_id in tqdm(node_name_numerical_idx_chemicals.items()):

        #retrieve the graph structure
        molecule_graph_i = dict_node_graph_sorted[node_id]

        if node_id == 0:

            for tens_atom_node in molecule_graph_i.x:

                edge_tensors_list_hyperedge.append(torch.tensor([[atom_counter,node_id]], dtype=torch.long))
                atom_counter+=1
                atom_idx_counter+=1
                node_features.append(tens_atom_node.reshape(1,-1))

            for e in range(molecule_graph_i.edge_index[0].shape[0]):
                edge_tensors_list_bonds.append(torch.tensor([[molecule_graph_i.edge_index[0][e],molecule_graph_i.edge_index[1][e]]], dtype=torch.long))
                edge_attribute_list_bonds.append(molecule_graph_i.edge_attr[e].reshape(1,-1))


        else:

            for tens_atom_node in molecule_graph_i.x:

                edge_tensors_list_hyperedge.append(torch.tensor([[atom_counter,node_id]], dtype=torch.long))
                atom_counter+=1
                node_features.append(tens_atom_node.reshape(1,-1))

            for e in range(molecule_graph_i.edge_index[0].shape[0]):
                edge_tensors_list_bonds.append(torch.tensor([[molecule_graph_i.edge_index[0][e] + atom_idx_counter,molecule_graph_i.edge_index[1][e]+atom_idx_counter]], dtype=torch.long))
                edge_attribute_list_bonds.append(molecule_graph_i.edge_attr[e].reshape(1,-1))

            atom_idx_counter += molecule_graph_i.x.shape[0]


    atom_node_features_tensor = torch.cat(node_features,dim=0)
    edge_tensors_hyperedge = torch.cat(edge_tensors_list_hyperedge,dim=0)
    edge_tensors_bonds = torch.cat(edge_tensors_list_bonds,dim=0)
    edge_attribute_bonds = torch.cat(edge_attribute_list_bonds,dim=0)

    ## add to the graph the hyperedge ('Atom','hyper','Chemical')
    Hetero_subgraph_comptoxAI['Atom','hyper','Chemical'].edge_index = edge_tensors_hyperedge.t().contiguous()
    Hetero_subgraph_comptoxAI['Atom','bond','Atom'].edge_index = edge_tensors_bonds.t().contiguous()
    Hetero_subgraph_comptoxAI['Atom','bond','Atom'].edge_attr = edge_attribute_bonds.contiguous()


    # # add number of atoms node
    Hetero_subgraph_comptoxAI['Atom'].num_nodes = atom_counter


    ## add the node features:
    # atom the new node features
    Hetero_subgraph_comptoxAI['Atom'].x = torch.tensor(atom_node_features_tensor,dtype=torch.long)

    # chemicals inizialized with 0
    Hetero_subgraph_comptoxAI['Chemical'].x = torch.zeros((Hetero_subgraph_comptoxAI['Chemical'].num_nodes,300),dtype=torch.float32)

    torch.save(Hetero_subgraph_comptoxAI,"./MolCLR/data/" + assay_repo + "/" + type_graph + "/Hetero_subgraph_comptoxAIs_hypernode.pt")
        
    print(Hetero_subgraph_comptoxAI)
    
    ####### add chemicallist
    add_chemlist_to_KG("./MolCLR/data/" + assay_repo + "/" + type_graph + "/Hetero_subgraph_comptoxAIs.pt",
                       chem_chemlist, dataset, node_name_numerical_idx_chemlist,node_name_numerical_idx_chemicals)
    add_chemlist_to_KG("./MolCLR/data/" + assay_repo + "/" + type_graph + "/Hetero_subgraph_comptoxAIs_hypernode.pt",
                       chem_chemlist, dataset, node_name_numerical_idx_chemlist,node_name_numerical_idx_chemicals)

def add_chemlist_to_KG(path_KG_to_load, chem_chemlist, dataset, node_name_numerical_idx_chemlist,node_name_numerical_idx_chemicals):
    
    Hetero_subgraph_comptoxAI = torch.load(path_KG_to_load)

    ######
    edge_tensors_list = []

    for index, row in tqdm(chem_chemlist.iterrows()):
        head = row['c.commonName']
        relation = row['type(r)']
        tail = dataset[dataset['cid'] == str(row['n.xrefPubchemCID'])]['name'].item()

        edge_tensors_list.append(torch.tensor([[node_name_numerical_idx_chemlist[head],node_name_numerical_idx_chemicals[tail]]], dtype=torch.long))

    edge_tensor = torch.cat(edge_tensors_list,dim=0)

    #######

    Hetero_subgraph_comptoxAI['ChemicalList',relation,'Chemical'].edge_index = edge_tensor.t().contiguous()

    Hetero_subgraph_comptoxAI['ChemicalList'].num_nodes = len(node_name_numerical_idx_chemlist)

    Hetero_subgraph_comptoxAI['ChemicalList'].x = torch.tensor(np.random.uniform(low=0,high=1,size=len(node_name_numerical_idx_chemlist)),dtype=torch.float32).reshape(-1,1)
        
    print(Hetero_subgraph_comptoxAI)
    torch.save(Hetero_subgraph_comptoxAI,path_KG_to_load.split(".pt")[0] + "_CL.pt")

def main():

    # create DB instance of ComptoxAI for connection
    db = GraphDB()
    
    # read the list of the selected assays
    with open("./list_tasks.pkl", "rb") as fp:
        list_task_tox21 = pickle.load(fp)

    # 1) base Knowledge Graph as a subgraph of ComptoxAI
    node_name_numerical_idx_chemicals = create_base_KG(db, list_task_tox21)

    # add the node features
    add_node_feature(list_task_tox21, node_name_numerical_idx_chemicals)

    # 2) Knowledge Graph with chemicals as hypernode made by atoms that are connected with bonds
    create_hierarchical_KG(list_task_tox21, node_name_numerical_idx_chemicals)


    # 3) create graph dataset for hERG, Neurotoxicity, CYP450 and Liver

    #### hERG
    hERG_mapped_comptoxAI = pd.read_excel('./MolCLR/data/hERG/hGER_df.xlsx')

    hERG_mapped_comptoxAI['cid'].value_counts()[hERG_mapped_comptoxAI['cid'].value_counts() == 2].index

    hERG_mapped_comptoxAI = hERG_mapped_comptoxAI.drop(325)
    hERG_mapped_comptoxAI = hERG_mapped_comptoxAI.drop_duplicates()

    hERG_mapped_comptoxAI['cid'] = hERG_mapped_comptoxAI['cid'].astype(str)

    
    chem_chemlist_herg = pd.read_csv("./MolCLR/data/hERG/to_add/chem_chemlist.csv")
    chem_gen_herg = pd.read_csv("./MolCLR/data/hERG/to_add/chem_gen.csv") 
    gen_gen_herg = pd.read_csv("./MolCLR/data/hERG/to_add/gen_gen.csv") 
    chem_assay_herg = pd.read_csv("./MolCLR/data/hERG/to_add/chem_assay.csv")

    # add herg assay
    datasets = [hERG_mapped_comptoxAI]
    assay_ML = ['hERG']

    type_r = []
    assay_name = []
    cids = []
        
    for data, data_name in zip(datasets, assay_ML):
        for index, row in data.iterrows():
            cids.append(row['cid'])
            assay_name.append(data_name)
            if row['target'] == 1:
                type_r.append('CHEMICALHASACTIVEASSAY')
            elif row['target'] == 0:
                type_r.append('CHEMICALHASINACTIVEASSAY')
            
    to_add_chem_assay = pd.DataFrame({'n.xrefPubchemCID':cids,'type(r)':type_r,'c.commonName':assay_name})
    chem_assay_herg = pd.concat([chem_assay_herg,to_add_chem_assay])


    total_dataset_prediction_assay = pd.read_excel("./tox21_chemicals_to_visualize.xlsx")
    total_dataset_prediction_assay['cid'] = total_dataset_prediction_assay['cid'].astype(str)

    hERG_mapped_comptoxAI_to_add = pd.concat([hERG_mapped_comptoxAI, total_dataset_prediction_assay]).drop_duplicates(subset=['cid'], keep='first')

    dict_assay_gene_involved = {'tox21-ahr-p1':["AHR"],
                                'tox21-ap1-agonist-p1':["FOS","JUN","FOSL2","FOSL1","JUNB","FOSB","JUND"],
                                'tox21-ar-bla-agonist-p1':["AR"],
                                'tox21-ar-bla-antagonist-p1':["AR"],
                                'tox21-ar-mda-kb2-luc-agonist-p1':["AR"],
                                'tox21-ar-mda-kb2-luc-agonist-p3':["AR"],
                                'tox21-ar-mda-kb2-luc-antagonist-p1':["AR"],
                                'tox21-ar-mda-kb2-luc-antagonist-p2':["AR"],
                                'tox21-are-bla-p1':["NFE2"],
                                'tox21-aromatase-p1':["CYP19A1"],
                                'tox21-car-agonist-p1':["NR1I3"],
                                'tox21-car-antagonist-p1':["NR1I3"],
                                'tox21-casp3-cho-p1':['CASP3'],
                                'tox21-casp3-hepg2-p1':['CASP3'],
                                'tox21-elg1-luc-agonist-p1':['ATAD5'],
                                'tox21-er-bla-agonist-p2':['ESR1'],
                                'tox21-er-bla-antagonist-p1':['ESR1'],
                                'tox21-er-luc-bg1-4e2-agonist-p2':['ESR1'],
                                'tox21-er-luc-bg1-4e2-agonist-p4':['ESR1'],
                                'tox21-er-luc-bg1-4e2-antagonist-p1':['ESR1'],
                                'tox21-er-luc-bg1-4e2-antagonist-p2':['ESR1'],
                                'tox21-erb-bla-antagonist-p1':["ESR2"],
                                'tox21-erb-bla-p1':["ESR2"],
                                'tox21-esre-bla-p1':["ATF6"],
                                'tox21-fxr-bla-agonist-p2':["NR1H4"],
                                'tox21-fxr-bla-antagonist-p1':["NR1H4"],
                                'tox21-gh3-tre-antagonist-p1':['THRB'],
                                'tox21-gr-hela-bla-agonist-p1':['NR3C1'],
                                'tox21-gr-hela-bla-antagonist-p1':['NR3C1'],
                                'tox21-h2ax-cho-p2':["H2AX"],
                                'tox21-hdac-p1':['HDAC1','HDAC2'],
                                'tox21-hre-bla-agonist-p1':['HIF1A'],
                                'tox21-hse-bla-p1':['HSPA5','HSPA8','HSPA1L','HSPA6','HSPA1A','HSPA4L','HSPA2','HSPA12A','HSPA13','HSPA1B','HSPA14','HSPA9','HSPA12B','HSPA4','HSPA7'],
                                'tox21-mitotox-p1':[],
                                'tox21-p53-bla-p1':['TP53'],
                                'tox21-ppard-bla-agonist-p1':['PPARD'],
                                'tox21-ppard-bla-antagonist-p1':['PPARD'],
                                'tox21-pparg-bla-agonist-p1':['PPARG'],
                                'tox21-pparg-bla-antagonist-p1':['PPARG'],
                                'tox21-pr-bla-agonist-p1':['PGR'],
                                'tox21-pr-bla-antagonist-p1':['PGR'],
                                'tox21-pxr-p1':['NR1I2'],
                                'tox21-rar-agonist-p1':['RARA','RARB','RARG'],
                                'tox21-rar-antagonist-p2':['RARA','RARB','RARG'],
                                'tox21-ror-cho-antagonist-p1':['RORC'],
                                'tox21-rt-viability-hepg2-p1':[],
                                'tox21-rxr-bla-agonist-p1':['RXRA'],
                                'tox21-sbe-bla-antagonist-p1':['SMAD2','SMAD3'],
                                'tox21-shh-3t3-gli3-antagonist-p1':['GLI1'],
                                'tox21-tshr-agonist-p1':['TSHR'],
                                'tox21-tshr-antagonist-p1':['TSHR'],
                                'tox21-vdr-bla-antagonist-p1':['VDR'],
                                'tox21-rt-viability-hek293-p1':[],
                                'tox21-sbe-bla-agonist-p1':['SMAD2','SMAD3'],
                                'tox21-tshr-wt-p1':['TSHR'],
                                'tox21-gh3-tre-agonist-p1':['THRB'],
                                'tox21-nfkb-bla-agonist-p1':['NFKB1','NFKB2','RELA','RELB','REL'],
                                'tox21-shh-3t3-gli3-agonist-p1':['GLI1'],
                                'tox21-vdr-bla-agonist-p1':['VDR'],
                            }

    assay_gen_herg = pd.DataFrame({'Assay':["hERG"], 'relation':['ASSAYTARGETGENE'], 'Gene':["KCNH2"]})

    add_dataset_to_graph(hERG_mapped_comptoxAI_to_add, chem_gen_herg, chem_assay_herg, chem_chemlist_herg, gen_gen_herg, assay_ML, dict_assay_gene_involved,assay_gen_herg,"to_add")

    ### Neurotoxicity

    neuro_BBB = pd.read_excel("./MolCLR/data/neuro/datasets_valid_and_splits/neuro_BBB_df.xlsx")
    neuro_NA = pd.read_excel("./MolCLR/data/neuro/datasets_valid_and_splits/neuro_NA_df.xlsx")
    neuro_NC = pd.read_excel("./MolCLR/data/neuro/datasets_valid_and_splits/neuro_NC_df.xlsx")
    neuro_NT = pd.read_excel("./MolCLR/data/neuro/datasets_valid_and_splits/neuro_NT_df.xlsx")

    all_neuro = pd.concat([neuro_BBB,neuro_NA,neuro_NC,neuro_NT]).drop_duplicates(subset=['cid'])
    all_neuro['cid'] = all_neuro['cid'].astype(str)


    chem_gen_neuro = pd.read_csv("./MolCLR/data/neuro/to_add/chem_gen.csv")
    chem_assay_neuro = pd.read_csv("./MolCLR/data/neuro/to_add/chem_assay.csv")
    gen_gen_neuro = pd.read_csv("./MolCLR/data/neuro/to_add/gen_gen.csv")
    chem_chemlist_neuro = pd.read_csv("./MolCLR/data/neuro/to_add/chem_chemlist.csv")

    total_dataset_prediction_assay = pd.read_excel("./tox21_chemicals_to_visualize.xlsx")
    total_dataset_prediction_assay['cid'] = total_dataset_prediction_assay['cid'].astype(str)

    all_neuro_to_add = pd.concat([all_neuro, total_dataset_prediction_assay]).drop_duplicates(subset=['cid'], keep='first')

    ## add the neuro assay
    datasets = [neuro_BBB,neuro_NA,neuro_NC,neuro_NT]
    datasets_name = ['neuro_BBB','neuro_NA','neuro_NC','neuro_NT']

    type_r = []
    assay_name = []
    cids = []
        
    for data, data_name in zip(datasets, datasets_name):
        for index, row in data.iterrows():
            cids.append(row['cid'])
            assay_name.append(data_name)
            if row['target'] == 1:
                type_r.append('CHEMICALHASACTIVEASSAY')
            elif row['target'] == 0:
                type_r.append('CHEMICALHASINACTIVEASSAY')
            
    to_add_chem_assay = pd.DataFrame({'n.xrefPubchemCID':cids,'type(r)':type_r,'c.commonName':assay_name})
    chem_assay = pd.concat([chem_assay,to_add_chem_assay])


    assay_ML = ['neuro_BBB','neuro_NA','neuro_NC','neuro_NT']
    assay_gen_neuro = pd.DataFrame({'Assay':[], 'relation':[], 'Gene':[]})

    add_dataset_to_graph(all_neuro_to_add, chem_gen_neuro, chem_assay_neuro, chem_chemlist_neuro, gen_gen_neuro, assay_ML, dict_assay_gene_involved,assay_gen_neuro,"to_add")


    # CYP450
    
    CYP450_CYP1A2 = pd.read_excel("./MolCLR/data/CYP450/datasets_valid_and_splits/CYP450_CYP1A2_df.xlsx")
    CYP450_CYP2C19 = pd.read_excel("./MolCLR/data/CYP450/datasets_valid_and_splits/CYP450_CYP2C19_df.xlsx")
    CYP450_CYP2C9 = pd.read_excel("./MolCLR/data/CYP450/datasets_valid_and_splits/CYP450_CYP2C9_df.xlsx")
    CYP450_CYP2D6 = pd.read_excel("./MolCLR/data/CYP450/datasets_valid_and_splits/CYP450_CYP2D6_df.xlsx")
    CYP450_CYP3A4 = pd.read_excel("./MolCLR/data/CYP450/datasets_valid_and_splits/CYP450_CYP3A4_df.xlsx")

    all_CYP450 = pd.concat([CYP450_CYP1A2,
                            CYP450_CYP2C19,
                            CYP450_CYP2C9,
                            CYP450_CYP2D6,
                            CYP450_CYP3A4]).drop_duplicates(subset=['cid'])

    all_CYP450['cid'] = all_CYP450['cid'].astype(str)

    chem_gen = pd.read_csv("./MolCLR/data/CYP450/alone/chem_gen.csv")
    chem_assay = pd.read_csv("./MolCLR/data/CYP450/alone/chem_assay.csv")
    gen_gen = pd.read_csv("./MolCLR/data/CYP450/alone/gen_gen.csv")
    chem_chemlist = pd.read_csv("./MolCLR/data/CYP450/alone/chem_chemlist.csv")


    datasets = [CYP450_CYP1A2,CYP450_CYP2C19,CYP450_CYP2C9,CYP450_CYP2D6,CYP450_CYP3A4]
    datasets_name = ['CYP450_CYP1A2','CYP450_CYP2C19','CYP450_CYP2C9','CYP450_CYP2D6','CYP450_CYP3A4']

    type_r = []
    assay_name = []
    cids = []
        
    for data, data_name in zip(datasets, datasets_name):
        for index, row in data.iterrows():
            cids.append(row['cid'])
            assay_name.append(data_name)
            if row['target'] == 1:
                type_r.append('CHEMICALHASACTIVEASSAY')
            elif row['target'] == 0:
                type_r.append('CHEMICALHASINACTIVEASSAY')
            
    to_add_chem_assay = pd.DataFrame({'n.xrefPubchemCID':cids,'type(r)':type_r,'c.commonName':assay_name})
    chem_assay = pd.concat([chem_assay,to_add_chem_assay])


    assay_ML = ['CYP450_CYP1A2','CYP450_CYP2C9','CYP450_CYP2C19','CYP450_CYP2D6','CYP450_CYP3A4']
    assay_gen_CYP450 = pd.DataFrame({'Assay':['CYP450_CYP1A2','CYP450_CYP2C9','CYP450_CYP2C19','CYP450_CYP2D6','CYP450_CYP3A4'], 
                                    'relation':['ASSAYTARGETGENE','ASSAYTARGETGENE','ASSAYTARGETGENE','ASSAYTARGETGENE','ASSAYTARGETGENE'], 
                                    'Gene':['CYP1A2','CYP2C9','CYP2C19','CYP2D6','CYP3A4']})
    

    chem_gen_CYP450 = pd.read_csv("./MolCLR/data/CYP450/to_add/chem_gen.csv")
    chem_assay_CYP450 = pd.read_csv("./MolCLR/data/CYP450/to_add/chem_assay.csv")
    gen_gen_CYP450 = pd.read_csv("./MolCLR/data/CYP450/to_add/gen_gen.csv")
    chem_chemlist_CYP450 = pd.read_csv("./MolCLR/data/CYP450/to_add/chem_chemlist.csv")

    total_dataset_prediction_assay = pd.read_excel("./tox21_chemicals_to_visualize.xlsx")
    total_dataset_prediction_assay['cid'] = total_dataset_prediction_assay['cid'].astype(str)

    all_CYP450_to_add = pd.concat([all_CYP450, total_dataset_prediction_assay]).drop_duplicates(subset=['cid'], keep='first')

    add_dataset_to_graph(all_CYP450_to_add, chem_gen_CYP450, chem_assay_CYP450, chem_chemlist_CYP450, gen_gen_CYP450, assay_ML, dict_assay_gene_involved,assay_gen_CYP450,"to_add")

if __name__ == '__main__':
    main()