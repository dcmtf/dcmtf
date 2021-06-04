## Multi-way Clustering and Discordance Analysis(DA) through Deep Collective Matrix Tri-Factorization (DCMTF)

Source code and data used in the experiments.

## 1) Clustering Performance

Commands to reproduce results of Table 1. Hyperparameters set as listed in Appendix F. 
In main_*.py, set `is_gpu` to `False` to run using CPU and change `gpu_id` as required when using GPU.

#### DCMTF

    `$ python -u main_dcmtf_clust.py <dataset_id> &> out.log`

#### CFRM: Collective Factorization of Related Matrices

	`$ python -u main_cfrm_clust.py <dataset_id>  &> out.log`

#### DFMF: Data Fusion by Matrix Factorization 
	
	`$python -u main_dfmf_clust.py <dataset_id>  &> out.log`

##### Parameters:

|<dataset_id>|Description|
| ------ | ------ |
| "wiki1" | Wikipedia dataset, 3 matrices, sample 1 - results shown in Table 1, sample used in Synchronizing case study |
| "wiki2" | Wikipedia dataset, 3 matrices, sample 2 - results shown in Appendix G |
| "wiki3" | Wikipedia dataset, 3 matrices, sample 3 - results shown in Appendix G |
| "wiki4" | Wikipedia dataset, 3 matrices, sample 4 - results shown in Appendix G |
| "genephene" | Cancer dataset, 5 matrices |
| "freebase" | Freebase dataset, 7 matrices |
| "pubmed" | PubMed dataset, 10 matrices |
| "pubmed_heuristic" | PubMed dataset, 10 matrices, sample used in HIN case study|

##### Outputs:

*  Clustering metrics in `out.log`
*  Entity representations, Entity cluster indicators, Cluster associations at `./out_clust/<dataset_id>/`


## 2) DA Case Study: Synchronizing Wikipedia Infoboxes

Steps to reproduce results of Table 2:

`Step 1:`  Obtain U, I and A using DCMTF/CFRM/DFMF for dataset "wiki1"

`Step 2:`  Perform DA using the corresponding ipy notebook. Open using Jupyter and run all cells. 

#### DCMTF + DA

1. `$ python -u main_dcmtf_clust.py "wiki1" &> out.log`
    
2. `da/wiki/"1 - DA wiki - DCMTF.ipynb"`

#### CFRM + DA

1. `$ python -u main_cfrm_clust.py "wiki1" &> out.log`

2. `da/wiki/"3 - DA wiki - CFRM.ipynb"`

#### DFMF + DA

1. `$python -u main_dfmf_clust.py "wiki1" &> out.log`

2. `da/wiki/"2 - DA wiki - DFMF.ipynb"`

##### *Note*:
To repeat this experiment for other wiki datasets *wiki2/wiki3/wiki4*: Run DCMTF for the required wiki[run_no] in `Step 1` and change the variable `run_no` accordingly to *2/3/4* in the ipython notebook before running them in `step 2`. 

##### Outputs:

*  Clustering metrics in `out.log`
*  Entity representations (U), Entity cluster indicators (I), Cluster associations (A) at `./out_clust/wiki1/`
*  Discordant Cluster Chain and the % entities found listed in Table 2


## 3) DA Case Study: Improving Network Representation Learning

Steps to obtain the 2 "cleaned" versions of the PubMed HIN from the `Original HIN` viz. `Rand-Cleaned` and `DA-Cleaned`:

1. Obtain U, I and A using DCMTF for dataset "pubmed_heuristic"
	
	`$ python -u main_dcmtf_clust.py "pubmed_heuristic" &> out.log`

2. Open using Jupyter and run all cells. Performs DA, obtains edge sets (i) E: from discordant chains, (ii) R: randomly selected

	`da/pubmed/"DA - HIN - step 1 - find high scoring and random cluster chains.ipynb"`

3. Open using Jupyter and run all cells. Filters E and R from `link.data` file of `Original HIN` to produce the cleaned up `link.data` files for `DA-Cleaned` and `Rand-Cleaned` in the folders `dcmtf/out_clust/pubmed_heuristic/version_2/cc/PubMed_da_link` and `dcmtf/out_clust/pubmed_heuristic/version_2/cc/PubMed_rand_link` respectively.
	
	`da/pubmed/"DA - HIN - step 2 - filter and obtain cleaned network.ipynb"`

4. Make copy of *Original HIN* data folder `PubMed_orig` to created data folders `PubMed_da` and `PubMed_rand` for *DA-Cleaned HIN* and *Rand-Cleaned HIN*. Replace `link.data` in these folders with the corresponding filtered versions `cc/PubMed_da_link/link.dat` and `cc/PubMed_rand_link/link.dat`

#### *Note:* 
The folder `/dcmtf/data_hin` contains the filtered version of the HINs `PubMed_orig`, `PubMed_da`, `PubMed_rand` used in our experimentation i.e. the output from the previous steps.


#### HIN2Vec, Metapath2Vec + DA

Steps to learn the HIN2Vec/Metapath2Vec embeddings for the HINs `PubMed_orig`, `PubMed_da`, `PubMed_rand` and obtain results on the two benchmark tasks (node classification and link prediction) shown in Table 3:

1. Copy the contents of `/data_hin/PubMed_orig/*` to `/HNE-master/Data/PubMed`
2. Do `cd HNE-master/Transform` and run `$sh transform.sh` 
3. Do `cd HNE-master/Model/HIN2Vec` and run `$sh run.sh` to learn the HIN2Vec embeddings
4. Do `cd HNE-master/Model/metapath2vec-ESim` and run `$sh run.sh` to learn the Metapath2Vec embeddings
5. Do `cd HNE-master/Evaluate` and run `$sh evaluate.sh` to obtain perform benchmark tasks and record results at: `/HNE-master/Data/PubMed/record.dat`
6. Repeat the above steps 1 to 5 for `PubMed_da` and `PubMed_rand`

More details about the baselines HIN2Vec, Metapath2Vec execution can be found [here](https://github.com/yangji9181/HNE)


## DCMTF - for arbitrary collection of matrices

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
- CFRM,DFMF: [Python27, preferably Anaconda distribution](https://docs.anaconda.com/anaconda/install/linux/#installation)
- DA: [NetworkX](https://networkx.org/)
- HIN2Vec: Python37, other details [here](https://github.com/yangji9181/HNE/tree/master/Model/HIN2Vec)
- Metapath2Vec: Python37, requires 2 external packages, details [here](https://github.com/yangji9181/HNE/tree/master/Model/metapath2vec-ESim) 


