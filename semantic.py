import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve,accuracy_score,matthews_corrcoef,f1_score, brier_score_loss,average_precision_score, roc_curve
from sklearn.calibration import calibration_curve

import random
from tqdm import tqdm
import os
import warnings
from typing import Dict, List, Optional

from torch import Tensor
import torch
from torch_geometric.nn import GATv2Conv,  Linear
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops


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

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, drop_ratio,norm_type,aggregation_type):
        super().__init__()
        
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.norm_type = norm_type
        
        # add for each edge a conv layer x number of layers
        self.convs = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            conv = HeteroConv_att({
                edge_type: GATv2Conv(-1, hidden_channels,add_self_loops=False)
                for edge_type in metadata[1]
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
        
        
    def forward(self, x_dict, edge_index_dict, chemical_label_task):
        
        total_att = []
        for layer in range(self.num_layers):
            
            h,att = self.convs[layer](x_dict, edge_index_dict)

            total_att.append(att)

            #h = self.convs[layer](h)
            
            for node_type, x in h.items():
                
                if self.norm_type=="batch" or self.norm_type=="layer":
                
                    if node_type=="Chemical":
                        h[node_type] = self.batch_norms_chemical[layer](x)
                    elif node_type=="Gene":
                        h[node_type] = self.batch_norms_gene[layer](x)
                    elif node_type=="Assay":
                        h[node_type] = self.batch_norms_assay[layer](x)


                if layer == self.num_layers - 1:
                    h[node_type] = F.dropout(x, self.drop_ratio, training=self.training)
                else:
                    h[node_type] = F.dropout(F.leaky_relu(x), self.drop_ratio, training=self.training)
            

        return self.pred_layer(h['Chemical'][chemical_label_task]),total_att
        
def train_semantic(list_task_tox21, node_name_numerical_idx_assay, node_name_numerical_idx_gene, node_name_numerical_idx_chemicals, Hetero_subgraph_comptoxAI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100

    for init_seed in [127, 128, 129, 130, 131]:

        for assay in list_task_tox21:

            print(assay)

            if os.path.exists("../results/semantic_gat/" + assay + "_" + str(init_seed) + ".csv"):
                continue

            torch.manual_seed(init_seed)
            random.seed(init_seed)
            np.random.seed(init_seed)

            ######## loading dataset
            dataset = pd.read_excel("./data/datasets_valid_and_splits/" + assay + "_df.xlsx")

            train_idx = np.load("./data/datasets_valid_and_splits/" + assay + "_train_idx_" + str(init_seed) + ".npy")
            valid_idx = np.load("./data/datasets_valid_and_splits/" + assay + "_valid_idx_" + str(init_seed) + ".npy")
            test_idx = np.load("./data/datasets_valid_and_splits/" + assay + "_test_idx_" + str(init_seed) + ".npy")

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
            'Gene': torch.tensor(list(node_name_numerical_idx_gene.values()))
                }

            subgraph_Hetero_subgraph_comptoxAI = Hetero_subgraph_comptoxAI.subgraph(subset_dict)


            #make undirected
            subgraph_Hetero_subgraph_comptoxAI = T.ToUndirected()(subgraph_Hetero_subgraph_comptoxAI)

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
                                    hidden_channels=parameters_grid['hidden_channels'],
                                    out_channels = 2,
                                    num_layers = parameters_grid['num_layers'],
                                    drop_ratio= parameters_grid['drop_ratio'],
                                    norm_type=parameters_grid['norm'],
                                    aggregation_type=parameters_grid['aggregation_node']).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=parameters_grid['learning_rate'], 
                                                weight_decay=parameters_grid['weight_decay'])
                criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(class_weight, device=device, dtype=torch.float32))


                best_val_metric = 0.0
                training_loss = []
                validation_loss = []

                for epoch in tqdm(range(num_epochs)):

                    model.train()
                    optimizer.zero_grad()

                    # model forward
                    out, att_wt = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
                                    subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                                    subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task)

                    # loss on the training set
                    loss = criterion(out[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask],
                                        subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].train_mask])

                    training_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    # validation
                    model.eval()
                    out, att_wv = model(subgraph_Hetero_subgraph_comptoxAI.x_dict, 
                                    subgraph_Hetero_subgraph_comptoxAI.edge_index_dict,
                                    subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_label_task)


                    loss = criterion(out[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask],
                                        subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask])

                    validation_loss.append(loss.item())

                    prediction_prob = F.softmax(out,dim=1)
                    prediction = prediction_prob.argmax(dim=1)
                    val_auc = roc_auc_score(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask].cpu().detach().numpy(),
                                                prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].val_mask][:,1].cpu().detach().numpy())


                    if val_auc>best_val_metric:
                        best_val_metric = val_auc
                        torch.save(model.state_dict(), './MolCLR/ckpt/best_run_semantics_' + str(init_seed) + '/semantic_model_test_' + assay + '.pth')


            model.load_state_dict(torch.load('./MolCLR/ckpt/best_run_semantics_' + str(init_seed) + '/semantic_model_test_' + assay + '.pth', map_location=device))

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


            test_tpr, test_fpr, _ = roc_curve(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(), prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask][:,1].cpu().detach().numpy())
            test_prob_pred, test_real_prob = calibration_curve(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(), prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask][:,1].cpu().detach().numpy(),n_bins=10)
            test_probabilities = prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask][:,1].cpu().detach().numpy()
            test_prec, test_rec, _ = precision_recall_curve(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(), prediction_prob[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask][:,1].cpu().detach().numpy())
            test_confusion = confusion_matrix(subgraph_Hetero_subgraph_comptoxAI['Chemical'].chemical_classification_label[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy(),prediction[subgraph_Hetero_subgraph_comptoxAI['Chemical'].test_mask].cpu().detach().numpy())


            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_tpr.npy",test_tpr,allow_pickle=True)
            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_fpr.npy",test_fpr,allow_pickle=True)
            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_prob_pred.npy",test_prob_pred,allow_pickle=True)
            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_real_prob.npy",test_real_prob,allow_pickle=True)
            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_test_prob.npy",test_probabilities,allow_pickle=True)
            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_prec.npy",test_prec,allow_pickle=True)
            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_rec.npy",test_rec,allow_pickle=True)
            # np.save("./MolCLR/results/semantic_gat_" + str(init_seed) + "/" + assay + "_confusion.npy",test_confusion,allow_pickle=True)

            new_row = pd.Series({"hyperparameters_combination":best_hyperparams_semantic,
                                        "validation_wROCAUC":best_val_metric,
                                        "test_wROCAUC":test_auroc,
                                        "test_wPRAUC":test_aupr,
                                        "test_brier":test_brier,
                                        "test_mcc":test_mcc,
                                        "test_wF1":test_f1,
                                        "test_accuracy_score": test_acc,
                                        "validation_loss":validation_loss,
                                        "training_loss":training_loss})

            dataframe_results = pd.concat([dataframe_results,new_row.to_frame().T], ignore_index=True)

            dataframe_results.to_csv("./results/semantic_gat_" + str(init_seed) + "/" + assay + ".csv",index=False)

            return 1


def main():

    # read the graph, nodes dictionaries and list of assay
    Hetero_subgraph_comptoxAI = torch.load("./data/Hetero_subgraph_comptoxAIs.pt")

    with open("./data/node_name_numerical_idx_gene.pkl", "rb") as fp:
        node_name_numerical_idx_gene = pickle.load(fp)
        
    with open("./data/node_name_numerical_idx_chemicals.pkl", "rb") as fp:
        node_name_numerical_idx_chemicals = pickle.load(fp)
        
    with open("./data/node_name_numerical_idx_assay.pkl", "rb") as fp:
        node_name_numerical_idx_assay = pickle.load(fp)
        
        
    with open("./data/list_tasks.pkl", "rb") as fp:
        list_task_tox21 = pickle.load(fp)

    train_semantic(list_task_tox21, node_name_numerical_idx_assay, node_name_numerical_idx_gene, node_name_numerical_idx_chemicals, Hetero_subgraph_comptoxAI)

if __name__ == "__main__":
    main()