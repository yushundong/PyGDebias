# Contribute to PyGDebias

We greatly appreciate your support and contributions. Please follow the steps below to contribute to **Algorithms** or **Datasets**. For the mechani

## Contribute to Datasets
Provide your datasets at any possible formats, while a *dataloading()* function which processes your datasets and outputs with the same formats as defined below. The best dataset split rate would be 0.5/0.25/0.25.
- *adj (Torch.Tensor): The adjacent matrix which is represented by a sparse tensor. # (node_num，node_num)*
- *features (Torch.Tensor): The features of all nodes. # (node_num, feature_num)*
- *labels (Torch.Tensor): lables for all nodes. # (node_num)*
- *idx_train (Torch.Tensor): the index of nodes in train dataset. # (train_size)*
- *idx_val (Torch.Tensor): the index of nodes in validation dataset. # (val_size)*
- *idx_test (Torch.Tensor): the index of nodes in test dataset. # (test_size)*
- *sens (Torch.Tensor): the vector of the sensitive group.# (node_num)*
- *sens_idx (int): the index of the sensitive group. # (int)*



## Contribute to Algorithms
Provide your algorithm/model in a python class whose name should be exactly the model name. Your algorithm/model should at least contain two fucntions, ie,
- *fit (model specified args restricted to the outputs of the dataloading() function stated above) --> (none):  execute the training process for the initiated graph mining algorithm.*
- *predict (none) --> (any returns): evaluate the trained graph mining algorithm on the test set.This function should print as many metrics as possible.*
  
For each algorithm, you should provide a description of the model, a clarification of the *fit()* function including the inputs types and formats, and a clarification of the *predict()* function including printed metrics' types and return type, better under each function definition with the format of """xxx""".


## Note 
When pulling requests
- Add your new dataset into the *dataset* folder. 
- Add your *dataloading()* function with an independent python file in the main directory.
- Add your algorithms in the *PyGDebias* folder.

If you are new to contributing, check the [tutorial](https://github.com/firstcontributions/first-contributions).