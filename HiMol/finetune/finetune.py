import argparse
from cmath import inf
import yaml
from data_utils import MoleculeDataset
from torch.utils.data import DataLoader
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import SubsetRandomSampler
from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score,matthews_corrcoef,f1_score, brier_score_loss,average_precision_score, balanced_accuracy_score

import os
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.data import Data

def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# def eval(args, model, device, loader):
#     model.eval()
#     y_true = []
#     y_scores = []

#     for step, batch in enumerate(tqdm(loader, desc="Iteration")):
#         batch = batch.to(device)

#         with torch.no_grad():
#             pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


#         y_true.append(batch.y.view(pred.shape))
#         y_scores.append(pred)

#     y_true = torch.cat(y_true, dim = 0).cpu().numpy()
#     y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

#     #Whether y is non-null or not.
#     y = batch.y.view(pred.shape).to(torch.float64)
#     is_valid = y**2 > 0
#     #Loss matrix
#     loss_mat = criterion(pred.double(), (y+1)/2)
#     #loss matrix after removing null target
#     loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
#     loss = torch.sum(loss_mat)/torch.sum(is_valid)


#     roc_list = []
#     for i in range(y_true.shape[1]):
#         #AUC is only defined when there is at least one positive data.
#         if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
#             is_valid = y_true[:,i]**2 > 0
#             roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

#     if len(roc_list) < y_true.shape[1]:
#         print("Some target is missing!")
#         print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

#     eval_roc = sum(roc_list)/len(roc_list) #y_true.shape[1]

#     return eval_roc, loss

def molgraph_to_graph_data(batch):
    graph_data_batch = []
    for mol in batch:
        data = Data(x=mol.x, edge_index=mol.edge_index, edge_attr=mol.edge_attr, num_part=mol.num_part, y=mol.y)
        graph_data_batch.append(data)
    new_batch = Batch().from_data_list(graph_data_batch)
    return new_batch


def main():
    config = yaml.load(open("./config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # tox21
    # with open("../../MolCLR/data/tox21/list_tasks_assay_ML.pkl", "rb") as fp:
    #    list_tasks = pickle.load(fp)

    # herg
    # list_tasks = ["hERG"]

    # neuro
    #list_tasks = ['neuro_BBB','neuro_NA','neuro_NC','neuro_NT']

    # CYP450
    # list_tasks = ['CYP450_CYP1A2', 'CYP450_CYP2C9', 'CYP450_CYP2C19', 'CYP450_CYP2D6', 'CYP450_CYP3A4']

    # Liver
    list_tasks = ['Liver_2vs', 'Liver_21vs']

    for num_task, task_name in enumerate(list_tasks):

            # insert  a tab for test from here

            # if os.path.exists("./results_Liver/graph_structure_comptoxAI_" + str(config['seed']) + "/test_" + task_name + "_metrics.csv"):
            #     continue

            datadf = pd.read_excel("../../HiMol/data/Liver/datasets_valid_and_splits/" + task_name + "_df.xlsx")

            print("Running task: " + task_name + " - task number: " + str(num_task))

            torch.manual_seed(config['seed'])
            random.seed(config['seed'])
            np.random.seed(config['seed'])

            # parameters_grid = {'init_lr': [0.0005, 0.001],
            #                        'init_base_lr':[5e-5, 5e-4],
            #                        'drop_ratio':[0, 0.3],
            #                        'pred_n_layer':[1, 2],
            #                        'weight_decay':['1e-6','1e-5'],
            #                        'feat_dim':[512]}
        
            # grid_search = ParameterGrid(parameters_grid)

            # df = pd.DataFrame(columns=["hyperparameters_combination","validation_wROCAUC"])
            df = pd.DataFrame(columns=["hyperparameters_combination","test_wROCAUC","test_wPRAUC","test_brier","test_mcc","test_wF1","test_accuracy_score"])

            with open("../data/Liver/best_hyperparams_MolCLR_5_runs.pkl", "rb") as fp:   
                best_hyperparams_MolCLR_5_runs = pickle.load(fp)

            parameters_combination = best_hyperparams_MolCLR_5_runs[task_name]
            best_hyperparams_MolCLR_5_runs = eval(best_hyperparams_MolCLR_5_runs[task_name])
            best_hyperparams_MolCLR_5_runs['parameters_combination'] = parameters_combination

            # to here


        # for parameters_combination in tqdm(grid_search):
        #     config['init_lr'] = parameters_combination['init_lr']
        #     config['init_base_lr'] = parameters_combination['init_base_lr']
        #     config['pred_n_layer'] = parameters_combination['pred_n_layer']
        #     config['weight_decay'] = parameters_combination['weight_decay']
        #     config['drop_ratio'] = parameters_combination['drop_ratio']
        #     config['feat_dim'] = parameters_combination['feat_dim']

            
            config['init_lr'] = best_hyperparams_MolCLR_5_runs['init_lr']
            config['init_base_lr'] = best_hyperparams_MolCLR_5_runs['init_base_lr']
            config['pred_n_layer'] = best_hyperparams_MolCLR_5_runs['pred_n_layer']
            config['weight_decay'] = best_hyperparams_MolCLR_5_runs['weight_decay']
            config['drop_ratio'] = best_hyperparams_MolCLR_5_runs['drop_ratio']
            config['feat_dim'] = best_hyperparams_MolCLR_5_runs['feat_dim']

            dataset = MoleculeDataset(datadf)

            train_idx = np.load("../../HiMol/data/Liver/datasets_valid_and_splits/" + task_name + "_train_idx_" + str(config['seed']) + ".npy")
            valid_idx = np.load("../../HiMol/data/Liver/datasets_valid_and_splits/" + task_name + "_valid_idx_" + str(config['seed']) + ".npy")
            test_idx =  np.load("../../HiMol/data/Liver/datasets_valid_and_splits/" + task_name + "_test_idx_" + str(config['seed']) + ".npy")

            training_df = datadf.filter(items=train_idx, axis=0).reset_index(drop=True)
            class_weight = compute_class_weight('balanced',classes=np.unique(training_df['target']), y = training_df['target'])

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            g = torch.Generator()
            g.manual_seed(config['seed'])

            train_loader = DataLoader(
                dataset, batch_size=config['batch_size'], sampler=train_sampler, collate_fn=lambda x:x, num_workers=0, drop_last=False, worker_init_fn=seed_worker, generator=g 
                )

            valid_loader = DataLoader(
                dataset, batch_size=config['batch_size'], sampler=valid_sampler, collate_fn=lambda x:x, num_workers=0, drop_last=False,  worker_init_fn=seed_worker, generator=g
                )

            test_loader = DataLoader(
                dataset, batch_size=config['batch_size'], sampler=test_sampler, collate_fn=lambda x:x, num_workers=0, drop_last=False, worker_init_fn=seed_worker, generator=g
            )

            criterion = nn.CrossEntropyLoss(weight = torch.tensor(class_weight,device=device,dtype=torch.float32))
            

            #set up model
            model = GNN_graphpred(config['model']['num_layer'], config['model']['emb_dim'], JK = "last", drop_ratio = config['drop_ratio'], gnn_type = 'gine')

            model.from_pretrained("../saved_model/pretrain.pth")
        
            model.to(device)

            layer_list = []
            for name, param in model.named_parameters():
                if 'graph_pred_linear' in name:
                    #print(name, param.requires_grad)
                    layer_list.append(name)

            params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
            base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

            optimizer = torch.optim.Adam(
                [{'params': base_params, 'lr': config['init_base_lr']}, {'params': params}],
                config['init_lr'], weight_decay=eval(config['weight_decay'])
            )

            model_checkpoints_folder = "./saved_model/Liver/best_run_models_" + str(config['seed']) + "/"

            if not os.path.exists(model_checkpoints_folder):
                os.makedirs(model_checkpoints_folder)

            #### training
            n_iter = 0
            valid_n_iter = 0
            best_valid_loss = np.inf
            best_valid_cls = 0
            training_losses = []
            validation_losses = []
            best_epoch = 0

            patience = 5

            for epoch_counter in range(config['epochs']):
                train_loss = 0.0
                num_data = 0

                model.train()

                for bn, dat in enumerate(train_loader):

                    optimizer.zero_grad()

                    graph_batch = molgraph_to_graph_data(dat)
                    graph_batch = graph_batch.to(device)
            
                    pred = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)

                    loss = criterion(pred, graph_batch.y.flatten())

                    train_loss += loss.item() * graph_batch.y.size(0)
                    num_data += graph_batch.y.size(0)


                    loss.backward()

                    optimizer.step()
                    n_iter += 1
                
                train_loss /= num_data
                training_losses.append(train_loss)


                if epoch_counter % config['eval_every_n_epochs'] == 0:

                    #### validation
                    predictions = []
                    labels = []
                    with torch.no_grad():
                        model.eval()

                        valid_loss = 0.0
                        num_data = 0
                        for bn, dat in enumerate(valid_loader):

                            graph_batch = molgraph_to_graph_data(dat)
                            graph_batch = graph_batch.to(device)

                            pred = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)
                            loss = criterion(pred, graph_batch.y.flatten())

                            valid_loss += loss.item() * graph_batch.y.size(0)
                            num_data += graph_batch.y.size(0)


                            pred = F.softmax(pred, dim=-1)

                            predictions.extend(pred.cpu().detach().numpy())
                            labels.extend(graph_batch.y.cpu().flatten().numpy())

                        valid_loss /= num_data
                        validation_losses.append(valid_loss)
                    
                    model.train()


                    predictions = np.array(predictions)
                    labels = np.array(labels)
                    valid_cls = roc_auc_score(labels, predictions[:,1])

                    if (valid_cls > best_valid_cls) and (valid_cls - best_valid_cls >= 0.01):
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, task_name + '_model_test.pth'))

                        patience = 5

                    else:
                        patience-=1
                        if patience == 0:
                            break

            # test
            predictions = []
            labels = []
            with torch.no_grad():
                model.eval()

                for bn, dat in enumerate(test_loader):

                    graph_batch = molgraph_to_graph_data(dat)
                    graph_batch = graph_batch.to(device)

                    pred = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)
                    pred = F.softmax(pred, dim=-1)

                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(graph_batch.y.cpu().flatten().numpy())

            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            average_precision = average_precision_score(labels, predictions[:,1])
            brier_score = brier_score_loss(labels, predictions[:,1])
            matthews_corr = matthews_corrcoef(labels, np.round(predictions[:,1]))
            f1_sc = f1_score(labels, np.round(predictions[:,1]),average='weighted')
            accuracy = accuracy_score(labels, np.round(predictions[:,1]))

            # new_row = pd.Series({"hyperparameters_combination":parameters_combination,
            #                     "validation_wROCAUC":best_valid_cls})

            new_row = pd.Series({"hyperparameters_combination":best_hyperparams_MolCLR_5_runs['parameters_combination'],
                            "test_wROCAUC":roc_auc,
                            "test_wPRAUC":average_precision,
                            "test_brier":brier_score,
                            "test_mcc":matthews_corr,
                            "test_wF1":f1_sc,
                            "test_accuracy_score":accuracy})

            df =  pd.concat([df, new_row.to_frame().T], ignore_index=True)

            df.to_csv("./results_Liver/graph_structure_comptoxAI_" + str(config['seed']) + "/test_" + task_name + "_metrics.csv",index=False)

if __name__ == "__main__":
    main()
