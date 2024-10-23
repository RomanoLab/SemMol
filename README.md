# SemMol
Augmenting molecular structure representation learning using semantic biomedical knowledge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13946620.svg)](https://doi.org/10.5281/zenodo.13946620)

### Requirements
- Python ≥ 3.8.0;
- ```requirements.txt``` contains the Python packages requirements.

### Data
The data used are made available through the following [box folder](https://upenn.box.com/v/SemMoldatasets), where you can find: 
- ```data/``` contains pretraining dataset, the Knowledge Graphs created with the relative dictionary of entities and their ids, the classification datasets (```datasets_valid_and_splits/``` contais for each assay the tabular dataset with Smiles string, MACCS key, chemicals name and labels, and the training, validataion and test index for the 5 random runs), and ```tsne_2d_embeddings_all_chemicals_37tox21_emb.xlsx```, that is a dataframe containing chemical names, MACCS keys, physical properties and the 2D t-SNE projections for all the n = 8541 chemicals that belong to the set of the 37 Tox21 assays considered;
- ```ckpt/``` contains the pretrained GNN molecule encoder.

### Models training for toxicology predictions
- <b>Machine learning</b>: baseline ML models can be trained by running ```ML.py```. The results will be written in ```results/ML``` with a directory for each random runs (seed).
- <b>Finetune MolCLR</b>: MolCLR can be finetuned by running ```MolCLR/finetune.py```. The results will be written in ```results/graph_structure_comptoxAI``` with a directory for each random runs (seed).
- <b>Semantic GNN</b>: Semantic GNN model can be trained by running ```semantic.py```.  The results will be written in ```results/semantic_gat``` with a directory for each random runs (seed).
- <b>MolCLR+Sem</b>: MolCLR+Sem model can be trained by running ```semantic_and_MolCLR.py```.  The results will be written in ```results/semantic_and_graph``` with a directory for each random runs (seed).

### XAI
Explainability with GNNExplainer can be obtained for positive chemicals by running the ```explain.py``` script. The results will be written in ```results/gnn_xai``` with a directory for each random runs.
```python
python explain.py
``` 

### Evaluation
The ```evaluation.py``` script contains code for:
- compute pretrained embeddings for all the chemcials involved in the Tox21 assays considered, project them in 2D with t-SNE and colour them according to chemical and physical properties of the molecules (extracted from ComptoxAI or through puchem API);
- process the classification results by computing the mean classification metrics for each model and for each assay, to create a dataframe than can be used to compute the violin plot with the mean results computedall the assay and the heatmap with the single assay results.
- process the xai results, by thresholding the number of edges to keep, and create the images with the molecule graph and most important subgraph identified a specific compound in input.
```python
python evaluate.py
```
