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

`Step 2:`  Perform DA using the corresponding ipy notebook. Open using Jupyter and run all cells

#### DCMTF + DA

1. `$ python -u main_dcmtf_clust.py "wiki1" &> out.log`
    
2. `da/wiki/"1 - DA wiki - DCMTF.ipynb"`

#### CFRM + DA

1. `$ python -u main_cfrm_clust.py "wiki1" &> out.log`

2. `da/wiki/"3 - DA wiki - CFRM.ipynb"`

#### DFMF + DA

1. `$python -u main_dfmf_clust.py "wiki1" &> out.log`

2. `da/wiki/"2 - DA wiki - DFMF.ipynb"`

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

3. Open using Jupyter and run all cells. Filters E and R from `link.data` file of `Original HIN` to produce the `link.data` files for `DA-Cleaned` and `Rand-Cleaned` respectively. 
	
	`da/pubmed/"DA - HIN - step 2 - filter and obtain cleaned network.ipynb"`

4. Make copy of `Original HIN` data folder `PubMed_orig`, replace the filtered version of `link.data` to create "cleaned" data folders: `PubMed_da` and `PubMed_rand` 	


#### HIN2Vec, Metapath2Vec + DA

Steps to learn the HIN2Vec/Metapath2Vec embeddings for `PubMed_orig`, and `PubMed_da` and `PubMed_rand` obtain results on the two benchmark tasks (node classification and link prediction) shown in Table 3:

1. Copy the contents of `/data_hin/PubMed_orig/*` to `/HNE-master/Data/PubMed`
2. Run `$ sh HNE-master/Transform/transform.sh` 
3. Run `$ sh HNE-master/Model/HIN2Vec/run.sh` to learn the HIN2Vec embeddings
4. Run `$ sh HNE-master/Model/metapath2vec-ESim/run.sh` to learn the Metapath2Vec embeddings
5. Run `$ sh HNE-master/Evaluate/evaluate.sh` to obtain perform benchmark tasks and record results at: `/HNE-master/Data/PubMed/record.dat`
6. Repeat the above steps 1 to 5 for `PubMed_da` and `PubMed_rand`

More details about the baselines HIN2Vec, Metapath2Vec execution can be found [here](https://github.com/yangji9181/HNE)

## Prerequisites
- DCMTF: [Python37, preferably Anaconda distribution](https://docs.anaconda.com/anaconda/install/linux/#installation)
- CFRM,DFMF: [Python27, preferably Anaconda distribution](https://docs.anaconda.com/anaconda/install/linux/#installation)
- DA: [NetworkX](https://networkx.org/)
- HIN2Vec: Python37, other details [here](https://github.com/yangji9181/HNE/tree/master/Model/HIN2Vec)
- Metapath2Vec: Python37, requires 2 external packages, details [here](https://github.com/yangji9181/HNE/tree/master/Model/metapath2vec-ESim) 


