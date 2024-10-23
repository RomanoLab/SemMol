from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
import random
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score, brier_score_loss,average_precision_score, accuracy_score
import os

def compute_metrics(model, X_val, X_test, validation_set, test_set, dataframe_results, classifier, parameters_grid):

    # make prediction
    y_pred_valid = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # compute score from prediction
    valid_metrics_acc = accuracy_score(validation_set['target'], y_pred_valid)
    test_metrics_acc  = accuracy_score(test_set['target'], y_pred_test)
    
    valid_metrics_mcc = matthews_corrcoef(validation_set['target'], y_pred_valid)
    test_metrics_mcc  = matthews_corrcoef(test_set['target'], y_pred_test)

    valid_metrics_wf1 = f1_score(validation_set['target'], y_pred_valid,average='weighted')
    test_metrics_wf1  = f1_score(test_set['target'], y_pred_test,average='weighted')
    

    # make prediction with probabilities
    y_pred_valid_prob = model.predict_proba(X_val)
    y_pred_test_prob = model.predict_proba(X_test)

    # compute score from probabilities
    valid_metrics_auc = roc_auc_score(validation_set['target'], y_pred_valid_prob[:,1])
    test_metrics_auc  = roc_auc_score(test_set['target'], y_pred_test_prob[:,1])

    valid_metrics_aucpr = average_precision_score(validation_set['target'], y_pred_valid_prob[:,1])
    test_metrics_aucpr  = average_precision_score(test_set['target'], y_pred_test_prob[:,1])
         
    valid_metrics_brier = brier_score_loss(validation_set['target'], y_pred_valid_prob[:,1])
    test_metrics_brier  = brier_score_loss(test_set['target'], y_pred_test_prob[:,1])      
        

    # append a row on the dataframe
    new_row = pd.Series({"hyperparameters_combination":parameters_grid,
                         "classifier":classifier,
                        "validation_accuracy_score":valid_metrics_acc,
                            "validation_wROCAUC":valid_metrics_auc,
                            "validation_wPRAUC":valid_metrics_aucpr,
                            "validation_mcc":valid_metrics_mcc,
                            "validation_wF1":valid_metrics_wf1,
                            "validation_brier": valid_metrics_brier,
                         
                            "test_accuracy_score":test_metrics_acc,
                            "test_wROCAUC":test_metrics_auc,
                            "test_wPRAUC":test_metrics_aucpr,
                            "test_mcc":test_metrics_mcc,
                            "test_wF1":test_metrics_wf1,
                            "test_brier": test_metrics_brier})
    
    dataframe_results =   pd.concat([dataframe_results, new_row.to_frame().T], ignore_index=True)

    return dataframe_results
    

def main():
    with open("./data/list_tasks.pkl", "rb") as fp:
        list_task_tox21 = pickle.load(fp)
    
    rn = 127
    random.seed(rn)
    np.random.seed(rn)

    for assay in list_task_tox21:
        
        print(assay)
        
        if os.path.exists("./results/ML_" + str(rn) + "/" + assay  + "_dataframe_results.xlsx"):
            continue
        
        dataframe_results = pd.DataFrame(columns = ["hyperparameters_combination","classifier",
                                                "validation_accuracy_score","validation_wROCAUC","validation_wPRAUC","validation_mcc","validation_wF1","validation_brier",
                                                "test_accuracy_score","test_wROCAUC","test_mcc","test_wPRAUC","test_wF1","test_brier"])
        
        # read the dataset
        dataset = pd.read_excel("./data/datasets_valid_and_splits/" + assay + "_df.xlsx")
        
        del dataset['name']
        del dataset['smiles']
        
        # make the maccs as features 
        df_maccs_features = pd.DataFrame([list(x) for x in dataset['maccs']], columns=["maccs_" + str(i) for i in range(0,167)])
        del df_maccs_features['maccs_0']
        
        # and concat
        dataset = pd.concat([dataset,df_maccs_features.astype('int')],axis=1)
        
        del dataset['maccs']
        
        train_idx = np.load("./data/datasets_valid_and_splits/" + assay + "_train_idx_" + str(rn) + ".npy")
        valid_idx = np.load("./data/datasets_valid_and_splits/" + assay + "_valid_idx_" + str(rn) + ".npy")
        test_idx = np.load("./data/datasets_valid_and_splits/" + assay + "_test_idx_" + str(rn) + ".npy")
        
        training_dataset = dataset.loc[train_idx].reset_index(drop=True)
        validation_dataset = dataset.loc[valid_idx].reset_index(drop=True)
        test_dataset = dataset.loc[test_idx].reset_index(drop=True)
        
        # class weight for RF
        class_weight = compute_class_weight('balanced',classes=np.unique(training_dataset['target']), y = training_dataset['target'])
        dict_class_weight = {}
        for c in np.unique(training_dataset['target']):
            dict_class_weight[c] = class_weight[c]
                
        # class weight for xgb
        sample_weight_list = []
        for sample in training_dataset['target']:
            if sample==0:
                sample_weight_list.append(class_weight[0])
            elif sample==1:
                sample_weight_list.append(class_weight[1])
        
        
        for classifier in ['random_forest','xgboost_tree']:
            
            if classifier == 'random_forest':
                parameters_grid = {'n_estimators': [50, 100],
                                'criterion': ['gini', 'entropy'],
                                'max_features':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                'min_samples_split': [2, 5, 8, 11, 14, 17, 20],
                                'min_samples_leaf': [1, 6, 11, 16, 21],
                                'bootstrap':[True, False],
                                }
                
                grid_search = ParameterGrid(parameters_grid)
                
                for parameters_combination in tqdm(grid_search):
                    model = RandomForestClassifier(random_state=rn, n_jobs= -1, 
                                                max_features = parameters_combination["max_features"],
                                                criterion = parameters_combination['criterion'],
                                                n_estimators = parameters_combination['n_estimators'],
                                                min_samples_leaf = parameters_combination['min_samples_leaf'],
                                                min_samples_split = parameters_combination['min_samples_split'],
                                                bootstrap = parameters_combination['bootstrap'],
                                                class_weight = dict_class_weight)
                    
                    model.fit(training_dataset.loc[:,~training_dataset.columns.isin(['target'])], training_dataset['target'])
                
                    dataframe_results  = compute_metrics(model, 
                                                        validation_dataset.loc[:,~validation_dataset.columns.isin(['target'])], 
                                                        test_dataset.loc[:,~test_dataset.columns.isin(['target'])], 
                                                        validation_dataset, test_dataset, dataframe_results, classifier, 
                                                        parameters_combination)

                
                
            elif classifier == "xgboost_tree":
                parameters_grid = {'n_estimators': [50, 100],
                                'max_depth': [1, 2, 3,4, 5, 6, 7 ,8, 9, 10],
                                'learning_rate':[0.1, 0.01, 0.001],
                                'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                'min_child_weight':[1, 6, 11, 16, 21]
                                }
                
                grid_search = ParameterGrid(parameters_grid)
                
                for parameters_combination in tqdm(grid_search):
                
                    model = xgb.XGBClassifier(objective="binary:logistic", random_state=rn, booster="gbtree", n_jobs = -1,
                                                learning_rate = parameters_combination['learning_rate'],
                                                max_depth = parameters_combination['max_depth'],
                                                min_child_weight = parameters_combination['min_child_weight'],
                                                n_estimators = parameters_combination['n_estimators'], 
                                                subsample= parameters_combination['subsample'])
                
                    model.fit(training_dataset.loc[:,~training_dataset.columns.isin(['target'])], training_dataset['target'], sample_weight=sample_weight_list )
        
                    dataframe_results  = compute_metrics(model, 
                                                        validation_dataset.loc[:,~validation_dataset.columns.isin(['target'])], 
                                                        test_dataset.loc[:,~test_dataset.columns.isin(['target'])], 
                                                        validation_dataset, test_dataset, dataframe_results, classifier, 
                                                        parameters_combination)
        
        
        dataframe_results.to_excel("./results/ML_" + str(rn) + "/" + assay  + "_dataframe_results.xlsx")





if __name__ == "__main__":
    main()
    