## Multi-way Clustering of Heterogeneous Data through Deep Collective Matrix Tri-Factorization (DCMTF)

Source code and data used in the experiments.

## 1) Clustering Performance

Commands to reproduce results of Table 1. Hyperparameters set as listed in Appendix B. 
In main_*.py, set `is_gpu` to `False` to run using CPU and change `gpu_id` as required when using GPU.

#### Augmented Multi-view

    `$ python -u main_dcmtf_clust.py <dataset_id> &> out.log`

#### Multi-view

    `$ python -u main_dcmtf_clust_multiview.py <dataset_id> &> out.log`

##### Parameters:

|<dataset_id>|Description|
| ------ | ------ |
| "wiki1" | Wikipedia dataset, 3 matrices |
| "genephene" | Cancer dataset, 5 matrices |
| "freebase" | Freebase dataset, 7 matrices |
| "pubmed" | PubMed dataset, 10 matrices |

##### Outputs:

*  Clustering metrics in `out.log`
*  Entity representations, Entity cluster indicators, Cluster associations at `./out_clust/<dataset_id>/`


## 2) Ablation Studies

`$ cd ablation_studies`

#### AE -> FFN. The autoencoder network is replaced by a feedforward network.

`$ python -u main_dcmtf_clust_abla.py "wiki1" "aec_remove" &> out.log`

#### C -> M. The clustering network is removed and k-means is performed on the representations learnt.

`$ python -u  main_dcmtf_clust_abla.py "wiki1" "ortho_remove" &> out.log`

#### 2->1-phase. raining is done in a single-phase instead of 2 phases

`$ python -u  main_dcmtf_clust_single_pass.py "wiki1" &> out.log`

##### Outputs:

*  Clustering metrics in `out.log`
*  Entity representations, Entity cluster indicators, Cluster associations at `./out_clust_abl/` or `./out_clust_single_pass/`


## 3) 20NG Dataset - Appendix E

`$ cd 20ng`

`$ python -u main_dcmtf_clust.py 20ng &> out.log`


## 4) DCMTF - for arbitrary collection of matrices

Code snippet to run DCMTF over an arbitrary collection of matrices. Refer to the `main_dcmtf_clust.py` for examples.


```
from src.dcmtf import dcmtf

#init DCMTF
dcmtf_instance = dcmtf(G, X_data, X_data_bef_pp, X_data_size_fac, X_meta, X_dtype,\
        k, kf, num_layers, e_actf, dict_num_clusters,\
        learning_rate, weight_decay, convg_thres, max_epochs,\
        is_pretrain, learning_rate_pretrain, weight_decay_pretrain, convg_thres_pretrain, max_epochs_pretrain,\
        mini_batch_size_frac, num_batches, dict_e_loss_weight, dict_loss_weight,\
        dict_e_size, y_val_dict,\
        is_gpu, is_train, is_load_init, is_rand_batch_train, \
        model_dir)
#fit
dcmtf_instance.fit()
#persist
dcmtf_instance.persist_out(out_dir)
```

##### Parameters:

| Parameter | Description |
| ------ | ------ |
| G | Entity-matrix graph as a dict with key:entity-id and value:list of associated matrix-ids |
| X_data | dict with key:matrix-id and value: matrix as numpy array |
| X_data_bef_pp | X_data before any pre-processing *(unused)*|
| X_data_size_fac | {} *(unused)*|
| X_meta | dict with key:matrix-id and value: (row entity-id, column entity-id) |
| X_dtype | dict with key:matrix-id and value:data-type: ["real"/"binary"] |
| k | Entity latent representation dimension |
| kf | None or fraction used to decide number of layers in the Autoenncoder as described in Appendix C.3 |
| num_layers | Number of Autoenncoder encoder/decoder layers: [0/1/2]. Applicable if kf == None | 
| e_actf | Encoder and Decoder activation function: ["tanh"/"sigma"/"relu"] |
| dict_num_clusters | dict with key:entity and value:number of clusters |
| learning_rate | Learning rate |
| weight_decay | Weight decay |
| convg_thres | Convergence threshold |
| max_epochs | Number of training iterations |
| is_pretrain | Pre-training done if True *(unused)*|
| learning_rate_pretrain | Learning rate to be used for pre-training *(unused)*|
| weight_decay_pretrain  |  Weight decay to be used for pre-training *(unused)*| 
| convg_thres_pretrain  | Convergence threshold to be used for pre-training *(unused)*| 
| max_epochs_pretrain | Number of iterations to be used for pre-training *(unused)*| 
| mini_batch_size_frac | None or fraction of instances to be used for mini-batching *(unused)*|
| num_batches | dict with key:entity and value:number of mini-batches | 
| dict_e_loss_weight | dict with key:entity and value: weight of the loss associated with the entity |
| dict_loss_weight | dict with key:loss_type and value: weight of the loss type. Eg: {"aec":1.0,"mat":1.0,"clust":1.0} |
| dict_e_size | dict with key:entity and value:number of entity instances | 
| y_val_dict | dict with key:entity and value: ground truth cluster labels or empty array if not assessed| 
| is_gpu | True to run on GPU, False to run on CPU |
| is_train | True to train from scratch, False load previously trained model and continue training |
| is_load_init | True to load the previously used model initializations, False otherwise |
| is_rand_batch_train | True | 
| model_dir | directory to store the model | 
| out_dir | directory to store the outputs: I, A, U | 


## Prerequisites
- DCMTF: [Python37, preferably Anaconda distribution](https://docs.anaconda.com/anaconda/install/linux/#installation)


