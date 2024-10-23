### Datasets creation
This subdirectory contains the following scripts for generating the datasets:

- <b>Pretraining dataset</b> (chemicals from ComptoxAI) and <b>classification datasets</b> (n=37 toxicology assays). Run the ```dataset_creation.py``` script to query 1) ComptoxAI chemicals to create the pretraining dataset and 2) ComptoxAI active and inactive chemicals for each assay;
```python
python dataset_creation.py
``` 

- <b>Pytorch geometric graphs</b>. Run the ```graph_dataset_creation.py``` script to first create a base KG with the extracted entities from ComptoxAI. Then two version of the heterogeneous graph are created. The first use as chemicals node features the chemicals embeddings extracted from the GNN pretrained encoder, while the second expands the graph by considering each chemicals as an hypernode, each connected to all the atoms that compose the molecule of the chemicals (belonging to the set of atoms entities).
```python
python graph_dataset_creation.py
``` 

In addition, ```list_tasks.pkl``` contains the name of the Tox21 tasks used, ```best_hyperparams_MolCLR_5_runs.pkl``` and ```best_hyperparams_semantic_5_runs.pkl``` contain the best hyperparameters that maximize the AUC on the validation set, i.e. the hyperparameters for MolCLR finetuning and Semantic GNN used to evaluate the models on the test set. 