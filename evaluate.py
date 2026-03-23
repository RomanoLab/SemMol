from tqdm import tqdm
import seaborn as sb
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tqdm import tqdm
import statsmodels.stats.multitest as sm
from scipy.stats import ttest_rel
import numpy as np
import yaml
from comptox_ai.db.graph_db import GraphDB
import pubchempy as pcp
from sklearn.manifold import TSNE

import sys
sys.path.append('../MolCLR/dataset_test')
from dataset_test import MolTestDataset
sys.path.append('../MolCLR/models/ginet_molclr')
from ginet_molclr import GINet

import torch
import torch_geometric

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import BondStereo 
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  

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
    smiles_data, labels, maccs_data, names = [], [], [], []
    smiles_transformed_as_a_graph = []
    for index, row in datadf.iterrows():

        smiles = row['smiles']
        label = row['target']
        maccs = row['maccs']
        name = row['name']
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and label != '':
            smiles_data.append(smiles)
            names.append(name)
            smiles_transformed_as_a_graph.append(index)
            maccs_data.append(maccs)
            if task == 'classification':
                labels.append(int(label))
            elif task == 'regression':
                labels.append(float(label))
            else:
                ValueError('task must be either regression or classification')

    return smiles_data, labels, maccs_data, smiles_transformed_as_a_graph, names

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

def compute_pretrained_embedding(list_task_tox21):
    total_dataset_prediction_assay = pd.DataFrame(columns=['name','maccs','smiles','target'])

    for assay in tqdm(list_task_tox21):
        #retrieve all the chemicals involved in the predictiont task
        dataset_prediction_assay = pd.read_excel("./data/datasets_valid_and_splits/" + assay + "_df.xlsx")
        total_dataset_prediction_assay = pd.concat([total_dataset_prediction_assay,dataset_prediction_assay],ignore_index=True)
        

    total_dataset_prediction_assay = total_dataset_prediction_assay[['name','smiles','maccs']].drop_duplicates()
    total_dataset_prediction_assay['target'] = [1 for i in range(len(total_dataset_prediction_assay))]

    all_dataset_tox21 = MolTestDataset(datadf=total_dataset_prediction_assay, target=1, task = "classification", task_name = "")
        
    #### load the pretrained model
    # read config file
    config_pretrain = yaml.load(open("../MolCLR/config_pretrain.yaml", "r"), Loader=yaml.FullLoader)

    # create a model instance
    model_pretrain = GINet(**config_pretrain["model"])

    # load the model pretrained weight
    model_pretrain.load_state_dict(torch.load('../MolCLR/ckpt/model.pth',map_location='cuda:0'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_pretrain.eval()
    tox21_dataset_embeddings_h_emb = []

    # compute the embedding for each chemical
    model_pretrain.to(device)
    for data,data_name in tqdm(zip(all_dataset_tox21,total_dataset_prediction_assay['name'])):    
        data = data[0]
        data.to(device)
        h_emb,h,out = model_pretrain(data)
        tox21_dataset_embeddings_h_emb.append(h_emb.detach().cpu())
        
    # apply tsne dimensionality reduction
    tsne_embedding_2d_emb = TSNE(n_components=2, learning_rate='auto', init='random', random_state=127).fit_transform(tox21_dataset_embeddings_h_emb)

    # make a df for the plots
    df_plot_emb = pd.DataFrame({"tsne-0":tsne_embedding_2d_emb[:,0],"tsne-1":tsne_embedding_2d_emb[:,1],'name':total_dataset_prediction_assay['name']})

    return df_plot_emb, total_dataset_prediction_assay

def enrich_pretrained_embedding(db, df_plot_emb, total_dataset_prediction_assay):

    molecular_weight = []
    mol_dict = {}
    initial_query = "MATCH (n:Chemical {commonName: "
    final_query = "}) RETURN n"
    tpsa_list = []
    complexity_list = []
    dataset_maccs = pd.DataFrame(columns=["maccs_" + str(i) for i in range(1,167)] + ['name'])

    for index,row in tqdm(df_plot_emb.iterrows()):

        ####### query comptoxai to recover the molecular weight from ComptoxAI
        total_query = initial_query + '"' + row['name'] + '"' + final_query
        node_entity = db.run_cypher(total_query)
        
        if len(node_entity)>0:
            if ('molWeight' in node_entity[0]['n'].keys()):
                if node_entity[0]['n']['molWeight']!="":
                    molecular_weight.append(float(node_entity[0]['n']['molWeight']))
                    mol_dict[row['name']] = float(node_entity[0]['n']['molWeight'])
                else:
                    # search with api if molweight not present in comptoxai
                    resuls_prop = pcp.get_properties('MolecularWeight',row['name'],'name')
                    if len(resuls_prop)>0:
                        if resuls_prop[0]['MolecularWeight']!="":
                            molecular_weight.append(float(resuls_prop[0]['MolecularWeight']))
                            mol_dict[row['name']] = float(resuls_prop[0]['MolecularWeight'])
                        else:
                            molecular_weight.append('FAIL')
                            mol_dict[row['name']] = 'FAIL'
                    else:
                        molecular_weight.append('FAIL')
                        mol_dict[row['name']] = 'FAIL'
            else:
                # search with api if molweight not attribute of comptoxai
                resuls_prop = pcp.get_properties('MolecularWeight',row['name'],'name')
                if len(resuls_prop)>0:
                    if resuls_prop[0]['MolecularWeight']!="":
                        molecular_weight.append(float(resuls_prop[0]['MolecularWeight']))
                        mol_dict[row['name']] = float(resuls_prop[0]['MolecularWeight'])
                    else:
                        molecular_weight.append('FAIL')
                        mol_dict[row['name']] = 'FAIL'
                else:
                    molecular_weight.append('FAIL')
                    mol_dict[row['name']] = 'FAIL'
                    
        else:
            molecular_weight.append('FAIL')
            mol_dict[row['name']] = 'FAIL'

        ####### query pubchem API to retrieve complexity and TPSA
        try:
            resuls_prop = pcp.get_properties(['TPSA','Complexity'],row['name'],'name')
            if len(resuls_prop)>0:
            
                    
                if resuls_prop[0]['TPSA']!="":
                    tpsa_list.append(float(resuls_prop[0]['TPSA']))
                else:
                    tpsa_list.append('FAIL')
                    
                if resuls_prop[0]['Complexity']!="":
                    complexity_list.append(float(resuls_prop[0]['Complexity']))
                else:
                    complexity_list.append('FAIL')
                    
            else:
                tpsa_list.append('FAIL')
                complexity_list.append('FAIL')
        except:
            tpsa_list.append('FAIL')
            complexity_list.append('FAIL')

        ####### maccs keys
        find_maccs = total_dataset_prediction_assay[total_dataset_prediction_assay['name'] == row['name']]
        dataset_maccs_new = pd.DataFrame([list(x) for x in find_maccs['maccs']]).astype(int)
        dataset_maccs_new.columns = ["maccs_" + str(i) for i in range(0,167)]
        del dataset_maccs_new['maccs_0']
        
        dataset_maccs_new['name'] = row['name']
        
        dataset_maccs = pd.concat([dataset_maccs,dataset_maccs_new])

    # assign the new columns and save the df for the plot
    df_plot_emb['xlogp'] = xlogp_list
    df_plot_emb['tpsa'] = tpsa_list
    df_plot_emb = df_plot_emb.merge(dataset_maccs,on="name")

    df_plot_emb.to_excel("./results/tsne_2d_embeddings_all_chemicals_tox21_emb.xlsx",index=False)

    return 1

def dimensionality_reduction_tsne_plot():

    df_plot_emb = pd.read_excel("./results/tsne_2d_embeddings_all_chemicals_tox21_emb.xlsx")

    df_plot_emb['MACCS key 162'] = df_plot_emb['maccs_162']
    df_plot_emb['MACCS key 49'] = df_plot_emb['maccs_49']
    df_plot_emb['MACCS key 165'] = df_plot_emb['maccs_165']

    fig, axs = plt.subplots(3, 2, figsize=(9.5, 12),layout='constrained') 

    count_r = 0
    count_c = 0

    for maccs,title in zip(['MACCS key 162','MACCS key 165','MACCS key 49',"Molecular Weight","tpsa","complexity"],
                        ["Aromatic","Presence of ring","Charge different from 0","Molecular weight (Da)","Polar surface area ($\AA$²)","Complexity"]):

        if "MACCS" in maccs:
        
            sb.scatterplot(data=df_plot_emb,x="tsne-0",y="tsne-1",hue=maccs,palette=["#47A0B3","#E2514A"],ax=axs[count_r,count_c])
            axs[count_r,count_c].set_title(title,fontdict = {'fontsize':22})
            #plt.legend([], [], frameon=False)
            #plt.axis('off')
            
        else:
            cc = sb.scatterplot(data=df_plot_emb[df_plot_emb[maccs]!='FAIL'],x="tsne-0",y="tsne-1",hue=maccs,hue_norm=(df_plot_emb[df_plot_emb[maccs]!='FAIL'][maccs].quantile(0.05),df_plot_emb[df_plot_emb[maccs]!='FAIL'][maccs].quantile(0.95)),palette='viridis',ax=axs[count_r,count_c])
            plt.legend([],[], frameon=False)
            norm = plt.Normalize(df_plot_emb[df_plot_emb[maccs]!='FAIL'][maccs].quantile(0.05), df_plot_emb[df_plot_emb[maccs]!='FAIL'][maccs].quantile(0.95))
            sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
            sm.set_array([])
            
            cc.get_legend().remove()
            #ax.figure.colorbar(sm)
            #plt.title("Complexity",fontsize=34)
            cbar = cc.figure.colorbar(sm,ax=axs[count_r,count_c])
            # plt.title("Molecular weight (Da)",size=34)
            #cbar.set_label(title, size=24,labelpad=10)  # Adjust label font size here
            cbar.ax.tick_params(labelsize=16) 
            cbar.ax.xaxis.set_label_position('top')
            axs[count_r,count_c].set_title(title,fontdict = {'fontsize':22})
            
        count_r+=1

        if count_r==3:
            count_r = 0
            count_c+=1
            
    fig.savefig("./figures/molecular_prop_vertical.png",dpi=300)

    return 1

def read_classification_results(list_task_tox21):

    df_plot = pd.DataFrame(columns =["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"] )
    df_plot_loss_positive = pd.DataFrame(columns=["classifier","fold","assay",'best_valid_loss','best_train_loss','% of positive','number of positive samples'])

    best_hyperparams_MolCLR_5_runs = {}
    best_hyperparams_semantic_5_runs = {}
    
    for assay in tqdm(list_task_tox21):
        print(assay)
        c = 1
        
        assay_ML_results_rf = []
        assay_ML_results_xgb = []
        results_ML = []
        results_MolCLR = []
        results_semantic = []
        results_semantic_and_MolCLR = []

        assay_MolCLR_results = []
        assay_semantic_results = []
        
        dataset_prediction_assay = pd.read_excel("./data/datasets_valid_and_splits/" + assay + "_df.xlsx")
        percentage_of_positive = (dataset_prediction_assay[dataset_prediction_assay['target']==1].shape[0]) / dataset_prediction_assay.shape[0] * 100
        number_sample = dataset_prediction_assay.shape[0]
        
        for seed in ["127","128","129","130","131"]:
            
            ########## ML
            fold_to_read_ML_assay = "./results/ML_" + seed + "/" + assay + "_dataframe_results.xlsx"

            results_fold_ML = pd.read_excel(fold_to_read_ML_assay)
            del results_fold_ML['Unnamed: 0']
            
            results_fold_ML['fold'] = [c for i in range(len(results_fold_ML))]
            results_fold_ML['assay'] = [assay for i in range(len(results_fold_ML))]
            results_fold_ML['% of positive'] = [percentage_of_positive for i in range(len(results_fold_ML))]
            
            
            results_ML.append(results_fold_ML)
            
            assay_ML_results_rf.append(results_fold_ML[results_fold_ML['classifier']=="random_forest"]['validation_wROCAUC'])
            assay_ML_results_xgb.append(results_fold_ML[results_fold_ML['classifier']=="xgboost_tree"]['validation_wROCAUC'])
            
            ########## MolCLR
            fold_to_read_MolCLR_assay = "./results/graph_structure_comptoxAI_" + seed + "/" + assay + "_metrics.csv"
            
            results_fold_MolCLR = pd.read_csv(fold_to_read_MolCLR_assay)
            
            results_fold_MolCLR['fold'] = [c for i in range(len(results_fold_MolCLR))]
            results_fold_MolCLR['classifier'] = ["MolCLR" for i in range(len(results_fold_MolCLR))]
            results_fold_MolCLR['assay'] = [assay for i in range(len(results_fold_MolCLR))]
            results_fold_MolCLR['% of positive'] = [percentage_of_positive for i in range(len(results_fold_MolCLR))]
            
            
            results_MolCLR.append(results_fold_MolCLR)
            assay_MolCLR_results.append(results_fold_MolCLR['validation_wROCAUC'])
            
            ########## semantic
            fold_to_read_semantic_assay = "./results/semantic_gat_" + seed + "/" + assay + ".csv"
            
            results_fold_semantic = pd.read_csv(fold_to_read_semantic_assay)
            
            results_fold_semantic['fold'] = [c for i in range(len(results_fold_semantic))]
            results_fold_semantic['classifier'] = ["mol emb + semantic" for i in range(len(results_fold_semantic))]
            results_fold_semantic['assay'] = [assay for i in range(len(results_fold_semantic))]
            results_fold_semantic['% of positive'] = [percentage_of_positive for i in range(len(results_fold_semantic))]
            
            results_semantic.append(results_fold_semantic)
            assay_semantic_results.append(results_fold_semantic['validation_wROCAUC'])
            
            # semantic and MolCLR
            fold_to_read_semantic_and_molecules_assay = "./MolCLR/results/semantic_and_graph_" + seed + "/" + assay + ".csv"
            
            results_fold_semantic_and_graph = pd.read_csv(fold_to_read_semantic_and_molecules_assay)
            
            results_fold_semantic_and_graph['fold'] = [c for i in range(len(results_fold_semantic_and_graph))]
            results_fold_semantic_and_graph['classifier'] = ["mol emb + semantic unified" for i in range(len(results_fold_semantic_and_graph))]
            results_fold_semantic_and_graph['assay'] = [assay for i in range(len(results_fold_semantic_and_graph))]
            results_fold_semantic_and_graph['% of positive'] = [percentage_of_positive for i in range(len(results_fold_semantic_and_graph))]
            
            results_semantic_and_MolCLR.append(results_fold_semantic_and_graph)
            c+=1
                    

        best_indx_rf = np.mean(assay_ML_results_rf,axis=0).argmax()
        best_indx_xgb = np.mean(assay_ML_results_xgb,axis=0).argmax()
        
        best_indx_molclr = np.mean(assay_MolCLR_results,axis=0).argmax()
        best_indx_semantic = np.mean(assay_semantic_results,axis=0).argmax()
        
        best_params_rf = results_fold_ML[results_fold_ML['classifier']=="random_forest"].reset_index().loc[best_indx_rf,'hyperparameters_combination']
        best_params_xgb = results_fold_ML[results_fold_ML['classifier']=="xgboost_tree"].reset_index().loc[best_indx_xgb,'hyperparameters_combination']

        best_params_molclr = results_fold_MolCLR.loc[best_indx_molclr,'hyperparameters_combination']
        best_params_semantic = results_fold_semantic.loc[best_indx_semantic,'hyperparameters_combination']
        
        best_hyperparams_MolCLR_5_runs[assay] = best_params_molclr
        best_hyperparams_semantic_5_runs[assay] = best_params_semantic
        
        metrics_test_rf = [results_ML[0][results_ML[0]['hyperparameters_combination'] == best_params_rf][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[1][results_ML[1]['hyperparameters_combination'] == best_params_rf][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[2][results_ML[2]['hyperparameters_combination'] == best_params_rf][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[3][results_ML[3]['hyperparameters_combination'] == best_params_rf][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[4][results_ML[4]['hyperparameters_combination'] == best_params_rf][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]]]
        
        metrics_test_rf = pd.concat([metrics_test_rf[0],metrics_test_rf[1],metrics_test_rf[2],metrics_test_rf[3],metrics_test_rf[4]])
        df_plot = pd.concat([df_plot,metrics_test_rf])
        
        
        metrics_test_xgb = [results_ML[0][results_ML[0]['hyperparameters_combination'] == best_params_xgb][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[1][results_ML[1]['hyperparameters_combination'] == best_params_xgb][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[2][results_ML[2]['hyperparameters_combination'] == best_params_xgb][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[3][results_ML[3]['hyperparameters_combination'] == best_params_xgb][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_ML[4][results_ML[4]['hyperparameters_combination'] == best_params_xgb][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]]]
        
        metrics_test_xgb = pd.concat([metrics_test_xgb[0],metrics_test_xgb[1],metrics_test_xgb[2],metrics_test_xgb[3],metrics_test_xgb[4]])
        df_plot = pd.concat([df_plot,metrics_test_xgb])

        
        fold_to_read_MolCLR_assay_test = "./results/graph_structure_comptoxAI_127/test_" + assay + "_metrics.csv"
        results_fold_MolCLR_test_0 = pd.read_csv(fold_to_read_MolCLR_assay_test)
        results_fold_MolCLR_test_0['classifier'] = "MolCLR"
        results_fold_MolCLR_test_0['assay'] = assay
        results_fold_MolCLR_test_0['fold'] = 1
        results_fold_MolCLR_test_0['% of positive'] = percentage_of_positive
        results_fold_MolCLR_test_0['number of positive samples'] = number_sample
        fold_to_read_MolCLR_assay_test = "./results/graph_structure_comptoxAI_128/test_" + assay + "_metrics.csv"
        results_fold_MolCLR_test_1 = pd.read_csv(fold_to_read_MolCLR_assay_test)
        results_fold_MolCLR_test_1['classifier'] = "MolCLR"
        results_fold_MolCLR_test_1['assay'] = assay
        results_fold_MolCLR_test_1['fold'] = 2
        results_fold_MolCLR_test_1['% of positive'] = percentage_of_positive
        results_fold_MolCLR_test_1['number of positive samples'] = number_sample
        fold_to_read_MolCLR_assay_test = "./results/graph_structure_comptoxAI_129/test_" + assay + "_metrics.csv"
        results_fold_MolCLR_test_2 = pd.read_csv(fold_to_read_MolCLR_assay_test)
        results_fold_MolCLR_test_2['classifier'] = "MolCLR"
        results_fold_MolCLR_test_2['assay'] = assay
        results_fold_MolCLR_test_2['fold'] = 3
        results_fold_MolCLR_test_2['% of positive'] = percentage_of_positive
        results_fold_MolCLR_test_2['number of positive samples'] = number_sample
        fold_to_read_MolCLR_assay_test = "./results/graph_structure_comptoxAI_130/test_" + assay + "_metrics.csv"
        results_fold_MolCLR_test_3 = pd.read_csv(fold_to_read_MolCLR_assay_test)
        results_fold_MolCLR_test_3['classifier'] = "MolCLR"
        results_fold_MolCLR_test_3['assay'] = assay
        results_fold_MolCLR_test_3['fold'] = 4
        results_fold_MolCLR_test_3['% of positive'] = percentage_of_positive
        results_fold_MolCLR_test_3['number of positive samples'] = number_sample
        fold_to_read_MolCLR_assay_test = "./results/graph_structure_comptoxAI_131/test_" + assay + "_metrics.csv"
        results_fold_MolCLR_test_4 = pd.read_csv(fold_to_read_MolCLR_assay_test)
        results_fold_MolCLR_test_4['classifier'] = "MolCLR"
        results_fold_MolCLR_test_4['assay'] = assay
        results_fold_MolCLR_test_4['fold'] = 5
        results_fold_MolCLR_test_4['% of positive'] = percentage_of_positive
        results_fold_MolCLR_test_4['number of positive samples'] = number_sample
            

        
        metrics_test_molclr = [results_fold_MolCLR_test_0[results_fold_MolCLR_test_0['hyperparameters_combination'] == best_params_molclr][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_MolCLR_test_1[results_fold_MolCLR_test_1['hyperparameters_combination'] == best_params_molclr][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_MolCLR_test_2[results_fold_MolCLR_test_2['hyperparameters_combination'] == best_params_molclr][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_MolCLR_test_3[results_fold_MolCLR_test_3['hyperparameters_combination'] == best_params_molclr][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_MolCLR_test_4[results_fold_MolCLR_test_4['hyperparameters_combination'] == best_params_molclr][["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]]]
        
        
        loss_positive_molclr = [results_fold_MolCLR_test_0[results_fold_MolCLR_test_0['hyperparameters_combination'] == best_params_molclr][["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_MolCLR_test_1[results_fold_MolCLR_test_1['hyperparameters_combination'] == best_params_molclr][["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_MolCLR_test_2[results_fold_MolCLR_test_2['hyperparameters_combination'] == best_params_molclr][["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_MolCLR_test_3[results_fold_MolCLR_test_3['hyperparameters_combination'] == best_params_molclr][["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_MolCLR_test_4[results_fold_MolCLR_test_4['hyperparameters_combination'] == best_params_molclr][["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']]]
        
        metrics_test_molclr = pd.concat([metrics_test_molclr[0],metrics_test_molclr[1],metrics_test_molclr[2],metrics_test_molclr[3],metrics_test_molclr[4]])
        df_plot = pd.concat([df_plot,metrics_test_molclr])
        
        loss_positive_molclr = pd.concat([loss_positive_molclr[0],loss_positive_molclr[1],loss_positive_molclr[2],loss_positive_molclr[3],loss_positive_molclr[4]])
        df_plot_loss_positive = pd.concat([df_plot_loss_positive,loss_positive_molclr])
        
        # val and train loss were originaly confused when saving the results

        fold_to_read_semantic_assay_test = "./results/semantic_gat_127/test_" + assay + ".csv"
        results_fold_semantic_test_0 = pd.read_csv(fold_to_read_semantic_assay_test)
        results_fold_semantic_test_0['classifier'] = "semantic"
        results_fold_semantic_test_0['assay'] = assay
        results_fold_semantic_test_0['fold'] = 1
        results_fold_semantic_test_0['% of positive'] = percentage_of_positive
        results_fold_semantic_test_0['number of positive samples'] = number_sample
        fold_to_read_semantic_assay_test = "./results/semantic_gat_128/test_" + assay + ".csv"
        results_fold_semantic_test_1 = pd.read_csv(fold_to_read_semantic_assay_test)
        results_fold_semantic_test_1['classifier'] = "semantic"
        results_fold_semantic_test_1['assay'] = assay
        results_fold_semantic_test_1['fold'] = 2
        results_fold_semantic_test_1['% of positive'] = percentage_of_positive
        results_fold_semantic_test_1['number of positive samples'] = number_sample
        fold_to_read_semantic_assay_test = "./results/semantic_gat_129/test_" + assay + ".csv"
        results_fold_semantic_test_2 = pd.read_csv(fold_to_read_semantic_assay_test)
        results_fold_semantic_test_2['classifier'] = "semantic"
        results_fold_semantic_test_2['assay'] = assay
        results_fold_semantic_test_2['fold'] = 3
        results_fold_semantic_test_2['% of positive'] = percentage_of_positive
        results_fold_semantic_test_2['number of positive samples'] = number_sample
        fold_to_read_semantic_assay_test = "./results/semantic_gat_130/test_" + assay + ".csv"
        results_fold_semantic_test_3 = pd.read_csv(fold_to_read_semantic_assay_test)
        results_fold_semantic_test_3['classifier'] = "semantic"
        results_fold_semantic_test_3['assay'] = assay
        results_fold_semantic_test_3['fold'] = 4
        results_fold_semantic_test_3['% of positive'] = percentage_of_positive
        results_fold_semantic_test_3['number of positive samples'] = number_sample
        fold_to_read_semantic_assay_test = "./results/semantic_gat_131/test_" + assay + ".csv"
        results_fold_semantic_test_4 = pd.read_csv(fold_to_read_semantic_assay_test)
        results_fold_semantic_test_4['classifier'] = "semantic"
        results_fold_semantic_test_4['assay'] = assay
        results_fold_semantic_test_4['fold'] = 5
        results_fold_semantic_test_4['% of positive'] = percentage_of_positive
        results_fold_semantic_test_4['number of positive samples'] = number_sample
            
        idx_best_loss_0 = np.argwhere(eval(results_fold_semantic_test_0['training_loss'][0]) == np.min(eval(results_fold_semantic_test_0['training_loss'][0])))[0][0]
        best_training_loss_0 = eval(results_fold_semantic_test_0['validation_loss'][0])[idx_best_loss_0]
        best_val_loss_0 = eval(results_fold_semantic_test_0['training_loss'][0])[idx_best_loss_0]
        results_fold_semantic_test_0['best_train_loss'] = best_training_loss_0
        results_fold_semantic_test_0['best_valid_loss'] = best_val_loss_0
        
        idx_best_loss_1 = np.argwhere(eval(results_fold_semantic_test_1['training_loss'][0]) == np.min(eval(results_fold_semantic_test_1['training_loss'][0])))[0][0]
        best_training_loss_1 = eval(results_fold_semantic_test_1['validation_loss'][0])[idx_best_loss_1]
        best_val_loss_1 = eval(results_fold_semantic_test_1['training_loss'][0])[idx_best_loss_1]
        results_fold_semantic_test_1['best_train_loss'] = best_training_loss_1
        results_fold_semantic_test_1['best_valid_loss'] = best_val_loss_1
        
        idx_best_loss_2 = np.argwhere(eval(results_fold_semantic_test_2['training_loss'][0]) == np.min(eval(results_fold_semantic_test_2['training_loss'][0])))[0][0]
        best_training_loss_2 = eval(results_fold_semantic_test_2['validation_loss'][0])[idx_best_loss_2]
        best_val_loss_2 = eval(results_fold_semantic_test_2['training_loss'][0])[idx_best_loss_2]
        results_fold_semantic_test_2['best_train_loss'] = best_training_loss_2
        results_fold_semantic_test_2['best_valid_loss'] = best_val_loss_2
        
        idx_best_loss_3 = np.argwhere(eval(results_fold_semantic_test_3['training_loss'][0]) == np.min(eval(results_fold_semantic_test_3['training_loss'][0])))[0][0]
        best_training_loss_3 = eval(results_fold_semantic_test_3['validation_loss'][0])[idx_best_loss_3]
        best_val_loss_3 = eval(results_fold_semantic_test_3['training_loss'][0])[idx_best_loss_3]
        results_fold_semantic_test_3['best_train_loss'] = best_training_loss_3
        results_fold_semantic_test_3['best_valid_loss'] = best_val_loss_3
        
        idx_best_loss_4 = np.argwhere(eval(results_fold_semantic_test_4['training_loss'][0]) == np.min(eval(results_fold_semantic_test_4['training_loss'][0])))[0][0]
        best_training_loss_4 = eval(results_fold_semantic_test_4['validation_loss'][0])[idx_best_loss_4]
        best_val_loss_4 = eval(results_fold_semantic_test_4['training_loss'][0])[idx_best_loss_4]
        results_fold_semantic_test_4['best_train_loss'] = best_training_loss_4
        results_fold_semantic_test_4['best_valid_loss'] = best_val_loss_4
        
        metrics_test_semantic = [results_fold_semantic_test_0[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_test_1[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_test_2[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_test_3[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_test_4[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]]]
        
        
        metrics_test_semantic = pd.concat([metrics_test_semantic[0],metrics_test_semantic[1],metrics_test_semantic[2],metrics_test_semantic[3],metrics_test_semantic[4]])
        df_plot = pd.concat([df_plot,metrics_test_semantic])
            
        
        loss_positive_semantic = [results_fold_semantic_test_0[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_semantic_test_1[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_semantic_test_2[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_semantic_test_3[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_semantic_test_4[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']]]
        
        loss_positive_semantic = pd.concat([loss_positive_semantic[0],loss_positive_semantic[1],loss_positive_semantic[2],loss_positive_semantic[3],loss_positive_semantic[4]])
        df_plot_loss_positive = pd.concat([df_plot_loss_positive,loss_positive_semantic])
        

        fold_to_read_semantic_graph_assay_test = "./results/semantic_and_graph_127/test_" + assay + ".csv"
        results_fold_semantic_graph_test_0 = pd.read_csv(fold_to_read_semantic_graph_assay_test)
        results_fold_semantic_graph_test_0['classifier'] = "semantic_graph"
        results_fold_semantic_graph_test_0['assay'] = assay
        results_fold_semantic_graph_test_0['fold'] = 1
        results_fold_semantic_graph_test_0['% of positive'] = percentage_of_positive
        results_fold_semantic_graph_test_0['number of positive samples'] = number_sample
        fold_to_read_semantic_graph_assay_test = "./results/semantic_and_graph_128/test_" + assay + ".csv"
        results_fold_semantic_graph_test_1 = pd.read_csv(fold_to_read_semantic_graph_assay_test)
        results_fold_semantic_graph_test_1['classifier'] = "semantic_graph"
        results_fold_semantic_graph_test_1['assay'] = assay
        results_fold_semantic_graph_test_1['fold'] = 2
        results_fold_semantic_graph_test_1['% of positive'] = percentage_of_positive
        results_fold_semantic_graph_test_1['number of positive samples'] = number_sample
        fold_to_read_semantic_graph_assay_test = "./MolCLR/results/semantic_and_graph_129/test_" + assay + ".csv"
        results_fold_semantic_graph_test_2 = pd.read_csv(fold_to_read_semantic_graph_assay_test)
        results_fold_semantic_graph_test_2['classifier'] = "semantic_graph"
        results_fold_semantic_graph_test_2['assay'] = assay
        results_fold_semantic_graph_test_2['fold'] = 3
        results_fold_semantic_graph_test_2['% of positive'] = percentage_of_positive
        results_fold_semantic_graph_test_2['number of positive samples'] = number_sample
        fold_to_read_semantic_graph_assay_test = "./MolCLR/results/semantic_and_graph_130/test_" + assay + ".csv"
        results_fold_semantic_graph_test_3 = pd.read_csv(fold_to_read_semantic_graph_assay_test)
        results_fold_semantic_graph_test_3['classifier'] = "semantic_graph"
        results_fold_semantic_graph_test_3['assay'] = assay
        results_fold_semantic_graph_test_3['fold'] = 4
        results_fold_semantic_graph_test_3['% of positive'] = percentage_of_positive
        results_fold_semantic_graph_test_3['number of positive samples'] = number_sample
        fold_to_read_semantic_graph_assay_test = "./MolCLR/results/semantic_and_graph_131/test_" + assay + ".csv"
        results_fold_semantic_graph_test_4 = pd.read_csv(fold_to_read_semantic_graph_assay_test)
        results_fold_semantic_graph_test_4['classifier'] = "semantic_graph"
        results_fold_semantic_graph_test_4['assay'] = assay
        results_fold_semantic_graph_test_4['fold'] = 5
        results_fold_semantic_graph_test_4['% of positive'] = percentage_of_positive
        results_fold_semantic_graph_test_4['number of positive samples'] = number_sample
            

            
            
        idx_best_loss_0 = np.argwhere(eval(results_fold_semantic_graph_test_0['training_loss'][0]) == np.min(eval(results_fold_semantic_graph_test_0['training_loss'][0])))[0][0]
        best_training_loss_0 = eval(results_fold_semantic_graph_test_0['validation_loss'][0])[idx_best_loss_0]
        best_val_loss_0 = eval(results_fold_semantic_graph_test_0['training_loss'][0])[idx_best_loss_0]
        results_fold_semantic_graph_test_0['best_train_loss'] = best_training_loss_0
        results_fold_semantic_graph_test_0['best_valid_loss'] = best_val_loss_0
        
        idx_best_loss_1 = np.argwhere(eval(results_fold_semantic_graph_test_1['training_loss'][0]) == np.min(eval(results_fold_semantic_graph_test_1['training_loss'][0])))[0][0]
        best_training_loss_1 = eval(results_fold_semantic_graph_test_1['validation_loss'][0])[idx_best_loss_1]
        best_val_loss_1 = eval(results_fold_semantic_graph_test_1['training_loss'][0])[idx_best_loss_1]
        results_fold_semantic_graph_test_1['best_train_loss'] = best_training_loss_1
        results_fold_semantic_graph_test_1['best_valid_loss'] = best_val_loss_1
        
        idx_best_loss_2 = np.argwhere(eval(results_fold_semantic_graph_test_2['training_loss'][0]) == np.min(eval(results_fold_semantic_graph_test_2['training_loss'][0])))[0][0]
        best_training_loss_2 = eval(results_fold_semantic_graph_test_2['validation_loss'][0])[idx_best_loss_2]
        best_val_loss_2 = eval(results_fold_semantic_graph_test_2['training_loss'][0])[idx_best_loss_2]
        results_fold_semantic_graph_test_2['best_train_loss'] = best_training_loss_2
        results_fold_semantic_graph_test_2['best_valid_loss'] = best_val_loss_2
        
        idx_best_loss_3 = np.argwhere(eval(results_fold_semantic_graph_test_3['training_loss'][0]) == np.min(eval(results_fold_semantic_graph_test_3['training_loss'][0])))[0][0]
        best_training_loss_3 = eval(results_fold_semantic_graph_test_3['validation_loss'][0])[idx_best_loss_3]
        best_val_loss_3 = eval(results_fold_semantic_graph_test_3['training_loss'][0])[idx_best_loss_3]
        results_fold_semantic_graph_test_3['best_train_loss'] = best_training_loss_3
        results_fold_semantic_graph_test_3['best_valid_loss'] = best_val_loss_3
        
        idx_best_loss_4 = np.argwhere(eval(results_fold_semantic_graph_test_4['training_loss'][0]) == np.min(eval(results_fold_semantic_graph_test_4['training_loss'][0])))[0][0]
        best_training_loss_4 = eval(results_fold_semantic_graph_test_4['validation_loss'][0])[idx_best_loss_4]
        best_val_loss_4 = eval(results_fold_semantic_graph_test_4['training_loss'][0])[idx_best_loss_4]
        results_fold_semantic_graph_test_4['best_train_loss'] = best_training_loss_4
        results_fold_semantic_graph_test_4['best_valid_loss'] = best_val_loss_4
            
        
        metrics_test_semantic_and_graph = [results_fold_semantic_graph_test_0[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_graph_test_1[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_graph_test_2[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_graph_test_3[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]],
                results_fold_semantic_graph_test_4[["classifier","test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier","fold","assay"]]]
        
        loss_positive_semantic_graph = [results_fold_semantic_graph_test_0[["classifier","fold","assay",'best_valid_loss',"best_train_loss","% of positive",'number of positive samples']],
                results_fold_semantic_graph_test_1[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_semantic_graph_test_2[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_semantic_graph_test_3[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']],
                results_fold_semantic_graph_test_4[["classifier","fold","assay",'best_valid_loss','best_train_loss',"% of positive",'number of positive samples']]]
            
                                        
        metrics_test_semantic_and_graph = pd.concat([metrics_test_semantic_and_graph[0],metrics_test_semantic_and_graph[1],metrics_test_semantic_and_graph[2],metrics_test_semantic_and_graph[3],metrics_test_semantic_and_graph[4]])
        df_plot = pd.concat([df_plot,metrics_test_semantic_and_graph])
            
            
        loss_positive_semantic_graph = pd.concat([loss_positive_semantic_graph[0],loss_positive_semantic_graph[1],loss_positive_semantic_graph[2],loss_positive_semantic_graph[3],loss_positive_semantic_graph[4]])
        df_plot_loss_positive = pd.concat([df_plot_loss_positive,loss_positive_semantic_graph])
                        

        ## save the configuration of the best hyperparmeters
        with open("./data/best_hyperparams_MolCLR_5_runs.pkl", "wb") as fp:   
            pickle.dump(best_hyperparams_MolCLR_5_runs, fp)
            
        with open("./data/best_hyperparams_semantic_5_runs.pkl", "wb") as fp:   
            pickle.dump(best_hyperparams_semantic_5_runs, fp)

        return df_plot, df_plot_loss_positive

def violin_plot(df_plot, test_metric):

    sb.set_theme(font_scale=1.6,style="whitegrid")
    comparison = [('random_forest','xgboost_tree'),
              ('random_forest','MolCLR'),
              ('random_forest','semantic'),
              ('random_forest','semantic_graph'),
              ('xgboost_tree','MolCLR'),
              ('xgboost_tree','semantic'),
              ('xgboost_tree','semantic_graph'),
              ('MolCLR','semantic'),
              ('MolCLR','semantic_graph'),
              ('semantic','semantic_graph')]
    
    plt.figure(figsize=(10,5))
    sb.violinplot(data = df_plot, x = "classifier", y = test_metric, hue = "classifier")
    plt.ylabel(test_metric)
    plt.xlabel("")
    plt.xticks([0,1,2,3,4],['RF','XGB','MolCLR','Semantic\nGNN','MolCLR+Sem'])

    plt.savefig("./figures/violin_" + test_metric + ".pdf",bbox_inches='tight',dpi=300)

    print("t test")

    pval_to_correct = []
    for comp in comparison:
        test_results = ttest_rel(df_plot[df_plot['classifier']==comp[0] ][test_metric],df_plot[df_plot['classifier']==comp[1] ][test_metric])
        pval_to_correct.append(test_results.pvalue)
        pvalue_adjusted = sm.multipletests(pval_to_correct, alpha=0.05, method='hs')
        if test_results.pvalue<=0.05:
            print(comp[0] + " vs. " + comp[1])
            print(test_results)
            print("\n")

    print("\n adjusted \n")        

    pvalue_adjusted = sm.multipletests(pval_to_correct, alpha=0.05, method='hs')
    for adj_p, comp in zip(pvalue_adjusted[1],comparison):
        if adj_p <= 0.05:
            print(comp)
            print(adj_p)

def heatmap_single_task(df_plot,test_metric):
    
    df_mean = df_plot.groupby(['classifier',"assay"])[test_metric].mean().unstack().loc[['random_forest','xgboost_tree','MolCLR','semantic','semantic_graph'],:]
    plt.figure(figsize=(12, 5))
    axc = sb.heatmap(df_mean, annot=False, fmt=".2f", cmap='viridis')
    colorbar = axc.collections[0].colorbar
    colorbar.set_label('Mean ' + test_metric, fontsize=16)  # Set the label and font size

    plt.xlabel('')
    plt.ylabel('')
    plt.yticks(np.linspace(start=0.5,stop=4.5,num=5),['RF','XGB','MolCLR','Semantic GNN','MolCLR+Sem'],size=15)
    plt.xticks(np.linspace(start=0.5,stop=36.5,num=37), [col.split('tox21-')[1] for col in df_mean.columns],size=12)
    plt.tight_layout()
    axc.figure.axes[-1].tick_params(labelsize=16)

    plt.savefig("./figures/heatmap_" + test_metric + ".png",bbox_inches='tight',dpi=300)
    plt.show()

    return 1

def read_and_threshold_explanations(list_task_tox21):
    list_of_explain = []
    c=0

    for assay in list_task_tox21:
        datadf = pd.read_excel("./data/datasets_valid_and_splits/" + assay + "_df.xlsx")
        datadf_pos = datadf[datadf['target'] == 1]
        
        for index,row in datadf_pos.iterrows():
            original_compund_name = row['name']
            compund_name = original_compund_name.replace("'","")
            compund_name = compund_name.replace("/","")
            compund_name = compund_name.replace(" ","")
            compund_name = compund_name.replace('"','')
            compund_name = compund_name.replace(':','')

            try:
                explanation_default_params = torch.load("./results/gnn_xai/" + assay + "/" + compund_name + ".pth")
                list_of_explain.append(explanation_default_params)
                
                # graph molecule plot
                nxgraph_default  = torch_geometric.utils.to_networkx(explanation_default_params,to_undirected=False,edge_attrs=["edge_mask"])
                # masked with half of the edge graph molecule plot
                nxgraph_default_th_50  = torch_geometric.utils.to_networkx(explanation_default_params.threshold(threshold_type="topk",value=int(explanation_default_params.edge_mask.shape[0]/2)),to_undirected=True,edge_attrs=["edge_mask"])
                # RDkit molecules
                mol = Chem.MolFromSmiles(datadf_pos[datadf_pos['name']==original_compund_name]['smiles'].item())

                # find edge id
                unique_bid_50 = []
                their_value = []
                for ie, e in enumerate(list(nxgraph_default_th_50.edges(data="edge_mask"))):
                    if e[2]!=0:
                        bid = mol.GetBondBetweenAtoms(e[0],e[1]).GetIdx()
                        unique_bid_50.append(bid)
                        their_value.append(e[2])

                bndhighlights = {}
                for ie, e in enumerate(unique_bid_50):
                    bndhighlights[e] = (1.0, 0.5, 0.5)


                # draw with RDkit the graph molecule
                drawer = rdMolDraw2D.MolDraw2DCairo(600,400)
                dopts = drawer.drawOptions()
                dopts.additionalAtomLabelPadding = 0.05
                # to add all the atom labels
                for i in range(mol.GetNumAtoms()):
                    dopts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()
                    
                dopts.useBWAtomPalette()
                drawer.FinishDrawing()
                drawer.DrawMolecule(mol)
                png_data = drawer.GetDrawingText()

                with open('./results/gnn_xai/' + assay + "/" + compund_name + ".png", 'wb') as f:
                    f.write(png_data)

                # draw with RDKit the masked graph molecule
                drawer_masked = rdMolDraw2D.MolDraw2DCairo(600,400)
                dopts = drawer_masked.drawOptions()
                dopts.additionalAtomLabelPadding = 0.05
                # to add all the atom labels
                for i in range(mol.GetNumAtoms()):
                    dopts.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol()
                
                dopts.useBWAtomPalette()
                drawer_masked.DrawMolecule(mol,highlightAtoms=[],highlightBonds=unique_bid_50)
                drawer_masked.FinishDrawing()
                png_data_masked50 = drawer_masked.GetDrawingText()

                with open('./results/gnn_xai/' + assay + "/" + compund_name + "_m.png", 'wb') as f:
                    f.write(png_data_masked50)

            except:
                print(compund_name)

    return 1

def make_the_example_plot_explanations(chemical_to_find, assay_of_interest):

    fig = plt.figure(figsize=(3.5, 2.5))
    count = 1

    for assay in assay_of_interest:
        datadf = pd.read_excel("./MolCLR/data/datasets_valid_and_splits/" + assay + "_df.xlsx")
        datadf_pos = datadf[datadf['target'] == 1]
        
        datadf_pos = datadf_pos[datadf_pos['name'].isin(chemical_to_find)]
        
        for index,row in datadf_pos.iterrows():

            original_compund_name = row['name']
            compund_name = original_compund_name.replace("'","")
            compund_name = compund_name.replace("/","")
            compund_name = compund_name.replace(" ","")
            compund_name = compund_name.replace('"','')
            compund_name = compund_name.replace(':','')

               

            png_data = img.imread('./results/gnnxai/' + assay + "/" + compund_name + ".png")
            png_data_masked_50 = img.imread('./results/gnnxai/' + assay + "/" + compund_name + "_m.png")

                
            plt.imshow(png_data_masked_50) 
            plt.title(original_compund_name,fontsize=16)
            plt.axis('off')
    
def main():

    # create DB instance of ComptoxAI for connection
    db = GraphDB()

    with open("./data/list_tasks.pkl", "rb") as fp:
        list_task_tox21 = pickle.load(fp)

    ##### PRETRAINED CHEMICALS EMBEDDING

    #  compute the pretrained embeddings for the Tox21 chemicals
    df_plot_emb, total_dataset_prediction_assay = compute_pretrained_embedding(list_task_tox21)

    # enrich the pretrained chemicals embedding with chemical and physical properties
    enrich_pretrained_embedding(db, df_plot_emb, total_dataset_prediction_assay)

    # make the tsne plot
    dimensionality_reduction_tsne_plot()

    ##### CLASSIFICATION

    # read the results for the classification over 5 random rus and find the best hyperparameters' values
    df_plot, df_plot_loss_positive = read_classification_results(list_task_tox21)

    for test_metric in ["test_wROCAUC","test_wPRAUC","test_mcc","test_wF1","test_brier"]:
        # make the violion plots with statistical test for the mean perfomance over all the assay for each classifier
        violin_plot(df_plot,test_metric)

        # make the heatmap with the mean results for each single assay
        heatmap_single_task(df_plot,test_metric)

    ##### XAI

    # read the explanations results to make figure with the threshold
    read_and_threshold_explanations(list_task_tox21)

    # read the explanation results to evaluate for a specific compound and assay
    for chem_to_explain in [["Metobromuron"],['Monolinuron'],['Chlorbromuron'],['Linuron'], ['Resveratrol']]:
        make_the_example_plot_explanations(chem_to_explain,'tox21-hdac-p1')
        

if __name__ == "__main__":
    main()