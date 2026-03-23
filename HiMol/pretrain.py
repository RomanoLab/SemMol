import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings("ignore")

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import rdkit
import sys
import logging, time
from tqdm import tqdm
import numpy as np
from gnn_model import GNN
from decoder import Model_decoder  

sys.path.append('./util/')

from data_utils import *



lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def group_node_rep(node_rep, batch_size, num_part):
    group = []
    super_group = []
    # print('num_part', num_part)
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count:count + num_atom])
        super_group.append(node_rep[count + num_all -1])
        count += num_all
    return group, super_group


def train(model_list, train_loader, val_loader, optimizer_list, device, patience, tolerance, output_model_file):
    model, model_decoder = model_list

    model.train()
    model_decoder.train()

    best_valid_loss = np.inf

    if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
    for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):



        #batch内的每个item是MolTree类型
        batch_size = len(batch)

        graph_batch = molgraph_to_graph_data(batch)


        graph_batch = graph_batch.to(device)
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        num_part = graph_batch.num_part
        node_rep, super_node_rep = group_node_rep(node_rep, batch_size, num_part)

        loss, bond_if_auc, bond_if_ap, bond_type_acc, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(batch, node_rep, super_node_rep)

        optimizer_list.zero_grad()

        loss.backward()

        optimizer_list.step()

        if_auc += bond_if_auc
        if_ap += bond_if_ap
        type_acc += bond_type_acc
        a_type_acc += atom_type_acc
        a_num_rmse += atom_num_rmse
        b_num_rmse += bond_num_rmse

        if (step+1) % 50 == 0:
            if_auc = if_auc / 50 
            if_ap = if_ap / 50 
            type_acc = type_acc / 50 
            a_type_acc = a_type_acc / 50
            a_num_rmse = a_num_rmse / 50
            b_num_rmse = b_num_rmse / 50

            print('Batch:',step,'loss:',loss.item())
            print('Batch:',step,'if_auc:',if_auc)
            print('Batch:',step,'if_ap:',if_ap)
            print('Batch:',step,'type_acc:',type_acc)
            print('Batch:',step,'a_type_acc:',a_type_acc)
            print('Batch:',step,'a_num_rmse:',a_num_rmse)
            print('Batch:',step,'b_num_rmse:',b_num_rmse)
            if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0

        if (step+1) % 1000 == 0:
            print("==validation==")
            validation_loss = validate(model_list, val_loader, optimizer_list, device)

            if best_valid_loss - validation_loss > abs(tolerance):
                patience = patience
                # save the model weights
                best_valid_loss = validation_loss
                print(best_valid_loss)

                torch.save(model.state_dict(), output_model_file)
            else:
                patience-=1
                if patience == 0:
                    return "early stopping criteria"

    return ""

def validate(model_list, loader, optimizer_list, device):

    valid_loss = 0

    with torch.no_grad():
        model, model_decoder = model_list

        model.eval()
        model_decoder.eval()
        if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
            #batch内的每个item是MolTree类型
            batch_size = len(batch)

            graph_batch = molgraph_to_graph_data(batch)
            graph_batch = graph_batch.to(device)
            node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
            num_part = graph_batch.num_part
            node_rep, super_node_rep = group_node_rep(node_rep, batch_size, num_part)

            loss, bond_if_auc, bond_if_ap, bond_type_acc, atom_type_acc, atom_num_rmse, bond_num_rmse = model_decoder(batch, node_rep, super_node_rep)

            valid_loss += loss.item()

            if_auc += bond_if_auc
            if_ap += bond_if_ap
            type_acc += bond_type_acc
            a_type_acc += atom_type_acc
            a_num_rmse += atom_num_rmse
            b_num_rmse += bond_num_rmse

            if (step+1) % 20 == 0:
                if_auc = if_auc / 20 
                if_ap = if_ap / 20 
                type_acc = type_acc / 20 
                a_type_acc = a_type_acc / 20
                a_num_rmse = a_num_rmse / 20
                b_num_rmse = b_num_rmse / 20

                print('Batch:',step,'loss:',loss.item())
                print('Batch:',step,'if_auc:',if_auc)
                print('Batch:',step,'if_ap:',if_ap)
                print('Batch:',step,'type_acc:',type_acc)
                print('Batch:',step,'a_type_acc:',a_type_acc)
                print('Batch:',step,'a_num_rmse:',a_num_rmse)
                print('Batch:',step,'b_num_rmse:',b_num_rmse)
                if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0
    
    valid_loss /= step+1

    model.train()
    model_decoder.train()

    return valid_loss
           
def seed_worker(worker_id):
    worker_seed = torch.initial_seed()
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='./data/zinc/all.txt',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--output_model_file', type=str, default='./saved_model/pretrain.pth',
                        help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument("--hidden_size", type=int, default=512, help='hidden size')
    parser.add_argument("--seed", type=int, default=203, help='random seed')
    parser.add_argument("--patience", type=int, default=5, help='patience for early stopping')
    parser.add_argument("--tolerance", type=float, default=0.001, help='tolerance for early stopping')
    args = parser.parse_args()


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = MoleculeDataset(args.dataset)


    num_train = len(dataset)
    indices = list(range(num_train))

    valid_idx = np.load("./data/valid_idx.npy",allow_pickle=True)
    train_idx = list(set(indices).difference(set(valid_idx)))

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                       collate_fn=lambda x:x, drop_last=True, generator=g, worker_init_fn=seed_worker, sampler=train_sampler)

    valid_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                       collate_fn=lambda x:x, drop_last=True, generator=g, worker_init_fn=seed_worker, sampler=valid_sampler)


    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    model_decoder = Model_decoder(args.hidden_size, device).to(device)

    model_list = [model, model_decoder]
    optimizer = optim.Adam([{"params":model.parameters()},{"params":model_decoder.parameters()}], lr=args.lr, weight_decay=args.decay)


    for epoch in range(1, args.epochs + 1):
        print('====epoch',epoch)
        string_results = train(model_list, train_loader, valid_loader, optimizer, device, args.patience, args.tolerance, args.output_model_file)

        if string_results == "early stopping criteria":
            print("reached early stopping at epoch: ", epoch)
            break


if __name__ == "__main__":
    main()
