### MolCLR

![Diagram](figs/MolCLR.gif?raw=true)

This subdirectory contains all the scripts necessary to pretrain and finetune the MolCLR model, which code is taken by the [MolCLR repository](https://github.com/yuyangw/MolCLR/tree/master).

- ```/ckpt```: contains the Pytorch checkpoint for the MolCLR pretrained, finetuned, Semantic GNN and MolCLR+Sem models, obtained by evaluating the models on the validation set;
- ```/dataset```: contains scrips for creating the molecules dataset and dataloader for both pretraing (```/dataset/dataset```) and finetuning (```/dataset/datset_test```);
- ```/models```: contains the Pytorch model to implement MolCLR with the GIN encoder for both pretraining (```/models/ginet_molclr```) and finetuning (```/models/ginet_finetune``);
- ```/utils```: contains the NT-XENT;
- ```config_pretraing.yaml``` and ```config_finetune.yaml``` contain the hyperparameters for the pretraining and finetuning, respectively. Note that the values for the finetuning hyperparameters' grid search are included in the ```finetune.py``` script;
- ```pretrain.py``` is the script for pretraining the MolCLR model and save it to ```ckpt/comptoxai/GIN_node_masking```;
```python
python pretrain.py
``` 
- ```finetune.py``` is script to finetune the MolCLR model.
```python
python finetune.py
``` 
