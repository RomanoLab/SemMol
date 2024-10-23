import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
import random
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score,matthews_corrcoef,f1_score, brier_score_loss,average_precision_score
from dataset.dataset_test_modified import MolTestDatasetWrapper
from sklearn.model_selection import ParameterGrid


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    #print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_finetune.yaml', os.path.join(model_checkpoints_folder, 'config_finetune.yaml'))


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):
    def __init__(self, dataset, config,task_name):
        self.config = config
        self.device = self._get_device()
        self.task_name = task_name

        self.dataset = dataset

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device


    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data)  # [N,C]

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())
        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()


        if self.config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss(weight = torch.tensor(self.dataset.class_weight,device=self.device,dtype=torch.float32))
        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()
            

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d, __ in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin':
            from models.ginet_finetune_modified import GINet
            model = GINet(self.config['pred_n_layer'], self.config['drop_ratio'],self.config['feat_dim'],**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_finetune import GCN
            model = GCN(self.config['dataset']['task'], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                #print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=eval(self.config['weight_decay'])
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = "./ckpt/"

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0
        self.best_valid_cls = 0


        for epoch_counter in range(self.config['epochs']):
            for bn, dat in enumerate(train_loader):

                data = dat

                optimizer.zero_grad()

                data = data.to(self.device)
                #loss = self._step(model, data, data_maccs, n_iter)
                loss = self._step(model, data, n_iter)

                #if n_iter % self.config['log_every_n_steps'] == 0:
                    #self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    #print(epoch_counter, bn, loss.item())
                    

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > self.best_valid_cls:
                        # save the model weights
                        self.best_valid_cls = valid_cls
                        torch.save(model.state_dict(), model_checkpoints_folder + "best_run_models_" + str(init_seed) +  '/model_test_' + self.task_name + '.pth')
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_test.pth'))

                #self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                #print(valid_loss)
                valid_n_iter += 1
        
        self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = "./ckpt/model.pth"
            state_dict = torch.load(checkpoints_folder, map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, dat in enumerate(valid_loader):

                data = dat
                
                data = data.to(self.device)

                __, pred = model(data)
                #loss = self._step(model, data, data_maccs, bn)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            #if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
            #    mae = mean_absolute_error(labels, predictions)
            #    print('Validation loss:', valid_loss, 'MAE:', mae)
            #    return valid_loss, mae
            #else:
            #    rmse = mean_squared_error(labels, predictions, squared=False)
            #    print('Validation loss:', valid_loss, 'RMSE:', rmse)
            #    return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            #print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = "./ckpt/best_run_models_" + str(init_seed) + "/model_test_" + self.task_name + ".pth"
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, dat in enumerate(test_loader):

                data = dat
                
                data = data.to(self.device)

                __, pred = model(data)
                #loss = self._step(model, data, data_maccs, bn)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        #model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            #if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
            #    self.mae = mean_absolute_error(labels, predictions)
            #    print('Test loss:', test_loss, 'Test MAE:', self.mae)
            #else:
            #    self.rmse = mean_squared_error(labels, predictions, squared=False)
            #    print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            self.average_precision_score = average_precision_score(labels, predictions[:,1])
            self.brier_score_loss = brier_score_loss(labels, predictions[:,1])
            self.matthews_corrcoef = matthews_corrcoef(labels, np.round(predictions[:,1]))
            self.f1_score = f1_score(labels, np.round(predictions[:,1]), average="weighted")
            self.accuracy_score = accuracy_score(labels, np.round(predictions[:,1]))




def main(config,datadf,task_name):
    dataset = MolTestDatasetWrapper(config['batch_size'],config['seed'], datadf, task_name, **config['dataset'])
    fine_tune = FineTune(dataset, config, task_name)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return [fine_tune.best_valid_cls, fine_tune.roc_auc, fine_tune.average_precision_score, fine_tune.brier_score_loss, fine_tune.matthews_corrcoef, fine_tune.f1_score, fine_tune.accuracy_score]
    if config['dataset']['task'] == 'regression':
        #if config['task_name'] in ['qm7', 'qm8', 'qm9']:
        #    return fine_tune.mae
        #else:
        #    return fine_tune.rmse
        pass


if __name__ == "__main__":
    config = yaml.load(open("./config_finetune.yaml", "r"), Loader=yaml.FullLoader)

    
    config['dataset']['task'] = 'classification'
    config['dataset']['target'] = 1

    # read the list of thte tasks name
    with open("../data/list_tasks.pkl", "rb") as fp:
        list_task_tox21 = pickle.load(fp)

    for init_seed in [127,128,129,130,131]:

        config['seed'] = init_seed
        
        # iterate over tasks
        for num_task, task_name in enumerate(list_task_tox21):

            datadf = pd.read_excel("../data/datasets_valid_and_splits/" + task_name + "_df.xlsx")

            print("Running task: " + task_name + " - task number: " + str(num_task))

                    
            #print(config)

            torch.manual_seed(config['seed'])
            random.seed(config['seed'])
            np.random.seed(config['seed'])


            # overwrite config

            parameters_grid = {'init_lr': [0.0005, 0.001],
                            'init_base_lr':[5e-5, 5e-4],
                            'drop_ratio':[0, 0.3],
                            'pred_n_layer':[1, 2],
                            'weight_decay':['1e-6','1e-5'],
                            'feat_dim':[512]}

            grid_search = ParameterGrid(parameters_grid)

            df = pd.DataFrame(columns=["hyperparameters_combination","validation_wROCAUC",
                                                                    "test_wROCAUC","test_wPRAUC","test_brier","test_mcc","test_wF1","test_accuracy_score"])
            
            for parameters_combination in tqdm(grid_search):
                config['init_lr'] = parameters_combination['init_lr']
                config['init_base_lr'] = parameters_combination['init_base_lr']
                config['pred_n_layer'] = parameters_combination['pred_n_layer']
                config['weight_decay'] = parameters_combination['weight_decay']
                config['drop_ratio'] = parameters_combination['drop_ratio']
                config['feat_dim'] = parameters_combination['feat_dim']


                #print(datadf)

                result = main(config,datadf,task_name)

                new_row = pd.Series({"hyperparameters_combination":parameters_combination,
                                "validation_wROCAUC":result[0],
                                "test_wROCAUC":result[1],
                                "test_wPRAUC":result[2],
                                "test_brier":result[3],
                                "test_mcc":result[4],
                                "test_wF1":result[5],
                                "test_accuracy_score":result[6]})
                
                df =  pd.concat([df, new_row.to_frame().T], ignore_index=True)

        
            df.to_csv("../results/graph_structure_comptoxAI_" + str(init_seed) + "/" + task_name + "_metrics.csv",index=False)
