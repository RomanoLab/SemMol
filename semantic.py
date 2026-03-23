import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, accuracy_score, matthews_corrcoef, f1_score, brier_score_loss, average_precision_score, roc_curve
from sklearn.calibration import calibration_curve
import random
from tqdm import tqdm
import os
import warnings
from typing import Dict, List, Optional

from torch import Tensor
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops
import torch_geometric

def load_data_graph_dict(graph_str, if_CL, dict_gene_str, dict_chem_str, dict_assay_str, dict_chemL_str, task_str):
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
    def __init__(self, convs: Dict[EdgeType, MessagePassing], aggr: Optional[str] = "sum"):
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
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(self, *args_dict, **kwargs_dict) -> Dict[NodeType, Tensor]:
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
                    args.append((value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                if not arg.endswith('_dict'):
                    raise ValueError(f"Keyword arguments need to end with '_dict'")
                arg = arg[:-5]
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None), value_dict.get(dst, None))

            if not has_edge_level_arg:
                continue

            out, attn_w = conv(*args, **kwargs, return_attention_weights=True)

            if dst not in out_dict:
                out_dict[dst] = [out]
            else:
                out_dict[dst].append(out)
                
            attn_w_dict[edge_type] = attn_w
                
        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)

        return out_dict, attn_w_dict
    
def run_semantic(train_test_type, assay_ML,graph_type, if_CL, results_fold, ckpt_fold, model_type):
    
    if model_type == "HiMol":
        model_folder_for_ckpt = './HiMol/finetune/saved_model/'
        model_folder_for_results = './HiMol/finetune/'
    else:
        model_folder_for_ckpt = './MolCLR/ckpt/comptoxai/GIN_node_masking/'
        model_folder_for_results = './MolCLR/'
    
    list_seeds = [127,128,129,130,131]
        
    for init_seed in list_seeds:


        #init_seed = 127
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_epochs = 100
        

        for assay in assay_ML:
            
            if len(assay_ML) !=1:
                if len(assay_ML) == 37:
                    assay_repo = assay_ML[0].split("-")[0]
                else:
                    assay_repo = assay_ML[0].split("_")[0]
            else:
                assay_repo = assay

            print(assay)

            if train_test_type == "test":
                if if_CL : 
                    with open("./" + model_type + "/data/" + assay_repo + "/" + graph_type + "/best_hyperparams_semantic_CL_5_runs.pkl", "rb") as fp:   
                        best_hyperparams_semantic_5_runs = pickle.load(fp)
                else:
                    with open("./" + model_type + "/data/" + assay_repo + "/" + graph_type + "/best_hyperparams_semantic_5_runs.pkl", "rb") as fp:   
                        best_hyperparams_semantic_5_runs = pickle.load(fp)

        #     if os.path.exists("./MolCLR/results_tox21/semantic_gat_CL/test_" + assay + "_" + str(init_seed) + ".csv"):
        #         continue

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

            #make undirected
            subgraph_Hetero_subgraph_comptoxAI = T.ToUndirected()(subgraph_Hetero_subgraph_comptoxAI)

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

            if train_test_type == "training":

                parameters_grid = {"hidden_channels":[128, 64],
                                  "num_layers":[1, 2],
                                  "drop_ratio":[0, 0.3],
                                  "learning_rate":[0.01, 0.001],
                                  "weight_decay":[1e-2, 1e-3, 1e-4],
                                  "norm":["","layer"],
                                  "aggregation_node":["sum","mean"]}

                grid_search = ParameterGrid(parameters_grid)


                for parameters_combination in tqdm(grid_search):
                    # create the model
                    model = HeteroGNN(subgraph_Hetero_subgraph_comptoxAI.metadata(),
                                        hidden_channels=parameters_combination['hidden_channels'],
                                        out_channels = 2,
                                        num_layers = parameters_combination['num_layers'],
                                        drop_ratio= parameters_combination['drop_ratio'],
                                        norm_type=parameters_combination['norm'],
                                        aggregation_type=parameters_combination['aggregation_node']).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=parameters_combination['learning_rate'], 
                                                    weight_decay=parameters_combination['weight_decay'])
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
                                    subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                                    subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task)

                        # loss on the training set
                        loss = criterion(out[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask],
                                        subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask])

                        validation_loss.append(loss.item())

                        loss.backward()
                        optimizer.step()

                        # validation
                        model.eval()
                        out, att_wv = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
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
                                    torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_CL_' + assay + '.pth')
                                else:
                                    torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_CL_' + assay + '.pth')
                            else:
                                if graph_type =="":
                                    torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_' + assay + '.pth')
                                else:
                                    torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_' + assay + '.pth')

                    if if_CL: 
                        if graph_type =="":
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_CL_' + assay + '.pth', map_location=device))
                        else:
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_CL_' + assay + '.pth', map_location=device))
                    else:
                        if graph_type =="":
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_' + assay + '.pth', map_location=device))
                        else:
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_' + assay + '.pth', map_location=device))
                        
                            
                        
                    # test at the end of the training
                    model.eval()
                    out, att_wte = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
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

                    new_row = pd.Series({"hyperparameters_combination":parameters_combination,
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
                        dataframe_results.to_csv(model_folder_for_results + results_fold  + "/semantic_gat_CL_" + str(init_seed) + "/" + assay + ".csv",index=False)
                    else:
                        dataframe_results.to_csv(model_folder_for_results + results_fold + "/" + graph_type + "/semantic_gat_CL_" + str(init_seed) + "/" + assay + ".csv",index=False)
                else:
                    if graph_type =="":
                        dataframe_results.to_csv(model_folder_for_results + results_fold + "/semantic_gat_" + str(init_seed) + "/" + assay + ".csv",index=False)
                    else:
                        dataframe_results.to_csv(model_folder_for_results + results_fold + "/" + graph_type + "/semantic_gat_" + str(init_seed) + "/" + assay + ".csv",index=False)


            elif train_test_type == "test":


                best_hyperparams_semantic = eval(best_hyperparams_semantic_5_runs[assay])

                #create the model
                model = HeteroGNN(subgraph_Hetero_subgraph_comptoxAI.metadata(),
                                        hidden_channels=best_hyperparams_semantic['hidden_channels'],
                                        out_channels = 2,
                                        num_layers = best_hyperparams_semantic['num_layers'],
                                        drop_ratio= best_hyperparams_semantic['drop_ratio'],
                                        norm_type=best_hyperparams_semantic['norm'],
                                        aggregation_type=best_hyperparams_semantic['aggregation_node']).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams_semantic['learning_rate'], 
                                                    weight_decay=best_hyperparams_semantic['weight_decay'])
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
                                    subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                                    subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task)

                    # loss on the training set
                    loss = criterion(out[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask],
                                        subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask])

                    validation_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    # validation
                    model.eval()
                    out, att_wv = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
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
                                torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_CL_test_' + assay + '.pth')
                            else:
                                torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_CL_test_' + assay + '.pth')
                        else:
                            if graph_type =="":
                                torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_test_' + assay + '.pth')
                            else:
                                torch.save(model.state_dict(), model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_test_' + assay + '.pth')

                            
                with torch.no_grad(): 
                     
                    if if_CL: 
                        if graph_type =="":
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_CL_test_' + assay + '.pth', map_location=device))
                        else:
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_CL_test_' + assay + '.pth', map_location=device))
                    else:
                        if graph_type =="":
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/semantic_model_test_' + assay + '.pth', map_location=device))
                        else:
                            model.load_state_dict(torch.load(model_folder_for_ckpt + ckpt_fold + '/best_run_semantics_' + str(init_seed) + '/' + graph_type + '/semantic_model_test_' + assay + '.pth', map_location=device))
                        
                        
                    # test at the end of the training
                    model.eval()
                    out, att_wte = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
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

                new_row = pd.Series({"hyperparameters_combination":best_hyperparams_semantic,
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

                # str(init_seed) + "/test_" + assay
                
                
                if if_CL:
                    if graph_type =="":
                        dataframe_results.to_csv(model_folder_for_results + results_fold  + "/semantic_gat_CL_" + str(init_seed) + "/test_" + assay + ".csv",index=False)
                    else:
                        dataframe_results.to_csv(model_folder_for_results + results_fold + "/" + graph_type + "/semantic_gat_CL_" + str(init_seed) + "/test_" + assay + ".csv",index=False)
                else:
                    if graph_type =="":
                        dataframe_results.to_csv(model_folder_for_results + results_fold + "/semantic_gat_" + str(init_seed) + "/test_" + assay + ".csv",index=False)
                    else:
                        dataframe_results.to_csv(model_folder_for_results + results_fold + "/" + graph_type + "/semantic_gat_" + str(init_seed) + "/test_" + assay + ".csv",index=False)

        



def main():

    # tox21
    for if_CL in [False, True]:
        load_data_results = load_data_graph_dict("./HiMol/data/tox21/Hetero_subgraph_comptoxAIs.pt", False, 
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

        run_semantic("test", list_task_tox21, "", if_CL, "results_tox21", "tox21",'MolCLR')
            

    # hERG
    list_task_tox21 = ['hERG']
    assay = "hERG"

    for if_CL in [False, True]:
        load_data_results = load_data_graph_dict("./HiMol/data/" + assay  + "/Hetero_subgraph_comptoxAIs.pt", if_CL, 
                                "./MolCLR/data/" + assay + "/node_name_numerical_idx_gene.pkl",
                                "./MolCLR/data/" + assay + "/node_name_numerical_idx_chemicals.pkl",
                                "./MolCLR/data/" + assay + "/node_name_numerical_idx_assay.pkl",
                                "./MolCLR/data/" + assay + "/node_name_numerical_idx_chemlist.pkl",
                                ['hERG'])
                
        Hetero_subgraph_comptoxAI = load_data_results[0]
        node_name_numerical_idx_chemicals = load_data_results[1]
        node_name_numerical_idx_gene = load_data_results[2]
        node_name_numerical_idx_assay = load_data_results[3]
        node_name_numerical_idx_chemlist = load_data_results[4]

        run_semantic("test", list_task_tox21, "", if_CL, "results_herg", "herg",'MolCLR')
            

    # neuro
    list_task_tox21 = ['neuro_BBB','neuro_NA','neuro_NC','neuro_NT']

    for if_CL in [False, True]:
        for assay in list_task_tox21:
            load_data_results = load_data_graph_dict("./MolCLR/data/neuro" + assay + "/Hetero_subgraph_comptoxAIs.pt", if_CL, 
                                "./MolCLR/data/neuro" + assay + "/node_name_numerical_idx_gene.pkl",
                                "./MolCLR/data/neuro" + assay + "/node_name_numerical_idx_chemicals.pkl",
                                "./MolCLR/data/neuro" + assay + "/node_name_numerical_idx_assay.pkl",
                                "./MolCLR/data/neuro" + assay + "/node_name_numerical_idx_chemlist.pkl",
                                list_task_tox21)
                
            Hetero_subgraph_comptoxAI = load_data_results[0]
            node_name_numerical_idx_chemicals = load_data_results[1]
            node_name_numerical_idx_gene = load_data_results[2]
            node_name_numerical_idx_assay = load_data_results[3]
            node_name_numerical_idx_chemlist = load_data_results[4]

            run_semantic("test", list_task_tox21, "", if_CL, "results_neuro", "neuro",'MolCLR')
            
            
            
    # CYP450        
    list_task_tox21 = ['CYP450_CYP1A2','CYP450_CYP2C9','CYP450_CYP2C19','CYP450_CYP2D6','CYP450_CYP3A4']

    for if_CL in [False, True]:
        for assay in list_task_tox21:
            load_data_results = load_data_graph_dict("./MolCLR/data/CYP450" + assay + "/Hetero_subgraph_comptoxAIs.pt", if_CL, 
                                    "./MolCLR/data/CYP450" + assay + "/node_name_numerical_idx_gene.pkl",
                                    "./MolCLR/data/CYP450" + assay + "/node_name_numerical_idx_chemicals.pkl",
                                    "./MolCLR/data/CYP450" + assay + "/node_name_numerical_idx_assay.pkl",
                                    "./MolCLR/data/CYP450" + assay + "/node_name_numerical_idx_chemlist.pkl",
                                    list_task_tox21)
                
            Hetero_subgraph_comptoxAI = load_data_results[0]
            node_name_numerical_idx_chemicals = load_data_results[1]
            node_name_numerical_idx_gene = load_data_results[2]
            node_name_numerical_idx_assay = load_data_results[3]
            node_name_numerical_idx_chemlist = load_data_results[4]

            run_semantic("test", list_task_tox21, "", if_CL, "results_CYP450", "CYP450",'MolCLR')
                
    # Liver
    list_task_tox21 = ['Liver_2vs']

    for if_CL in [False, True]:
        for assay in list_task_tox21:
            load_data_results = load_data_graph_dict("./MolCLR/data/Liver/" + assay + "/Hetero_subgraph_comptoxAIs.pt", if_CL, 
                                "./MolCLR/data/Liver" + "/" + assay + "/node_name_numerical_idx_gene.pkl",
                                "./MolCLR/data/Liver" + "/" + assay +"/node_name_numerical_idx_chemicals.pkl",
                                "./MolCLR/data/Liver" + "/" + assay + "/node_name_numerical_idx_assay.pkl",
                                "./MolCLR/data/Liver" + "/" + assay + "/node_name_numerical_idx_chemlist.pkl",
                                list_task_tox21)
                
            Hetero_subgraph_comptoxAI = load_data_results[0]
            node_name_numerical_idx_chemicals = load_data_results[1]
            node_name_numerical_idx_gene = load_data_results[2]
            node_name_numerical_idx_assay = load_data_results[3]
            node_name_numerical_idx_chemlist = load_data_results[4]

            run_semantic("test", list_task_tox21, "", if_CL, "results_Liver", "Liver",'MolCLR')
            

if __name__ == "__main__":
    main()