#!/usr/bin/env python
# coding: utf-8


import os

import sys
import numpy as np
import pickle as pkl
import pandas as pd
import traceback
import random

import matplotlib.pyplot as plt

import torch
from torch import nn

from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans


from src.dcmtf_single_pass import dcmtfsinglepass
import importlib

# In[5]:


from src.vae import vae
#import src.ffn_clust
from src.ffn_clust import ffnclust
from src.ffn_ucat import ffnucat

import pprint as pp

# In[6]:

import random

np.random.seed(0)
random.seed(0)


import time
time_start = time.time()

import bz2
import _pickle as cPickle

# Load any compressed pickle file
def decompress_pickle(file):
    assert file.endswith("pbz2")
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data,encoding='latin1')
    return data

def get_size_fac_pp(X_cg):
    #print("get_size_fac_pp - X.shape: ",X_cg.shape)
    #
    epsilon1 = 1
    epsilon2 = 0
    X_cg = X_cg + epsilon1
    #print("get_size_fac_pp - X.shape: ",X_cg.shape)
    X_cg_sum = np.sum(X_cg,axis=1)
    X_cg_size_fac = X_cg_sum / (np.median(X_cg_sum)+epsilon2)
    #print("get_size_fac_pp - X_cg_size_fac.shape: ",X_cg_size_fac.shape)
    #Eqn (4)
    X_cg_size_fac_diag = np.diag(1.0 / X_cg_size_fac)
    X_cg_size_fac_diag_mm_x = np.dot(X_cg_size_fac_diag,X_cg)
    X_cg_size_fac_diag_mm_x_log = np.log(X_cg_size_fac_diag_mm_x + 1)
    #print("get_size_fac_pp - X_cg_size_fac_diag_mm_x_log.shape: ",X_cg_size_fac_diag_mm_x_log.shape)
    pp_scaler = StandardScaler()
    X_cg_size_fac_diag_mm_x_log_std = pp_scaler.fit_transform(X_cg_size_fac_diag_mm_x_log)
    #
    #X_cg_size_fac = X_cg_size_fac - 1
    #
    return X_cg_size_fac, X_cg_size_fac_diag_mm_x_log_std


X_data_size_fac = {}


assert len(sys.argv) == 2 ,"Usage: python main_dcmtf_clust.py dataset_id"

print("cur_dir: ",os.getcwd())

dataset_id = sys.argv[1]
abl_category = "None"
# gpu_id = int(sys.argv[2])
# num_batches = int(sys.argv[3])
#is_rand_batch_train_temp = int(sys.argv[4])
print("abl_category: ",abl_category)
print("#")
assert abl_category in ["None"]

gpu_id = 0
is_gpu = True
# adr_data_version_name = None
# if len(sys.argv) == 6:
#     adr_data_version_name = sys.argv[5]
#    print("data_version_name: ",adr_data_version_name)
data_version_name = "2"

is_rand_batch_train = True
is_single_pass = True
print("#")
print("is_single_pass: ",is_single_pass)
print("#")
# if is_rand_batch_train_temp == 1:
#     is_rand_batch_train = True
# else:
#     is_rand_batch_train = False

assert dataset_id in ["wiki1","wiki2","wiki3","wiki4","pubmed", "freebase","genephene","pubmed_heuristic"],"Unknown dataset_id: "+str(dataset_id) #sample "wiki5" unused
print("dataset_id: ",dataset_id)

# if dataset_id in ["freebase"]:
#     assert not adr_data_version_name == None

data_dir = "./data/"
fname = data_dir + dataset_id + "/" 

if dataset_id == "wiki1":
   num_batches = 10
   fname += "dict_data_item_subj_v4_hidden_run1.pkl.p2.pbz2"
elif dataset_id == "wiki2":
   num_batches = 10
   fname += "dict_data_item_subj_v4_hidden_run2.pkl.p2.pbz2"
elif dataset_id == "wiki3":
   num_batches = 10
   fname += "dict_data_item_subj_v4_hidden_run3.pkl.p2.pbz2"
elif dataset_id == "wiki4":
   num_batches = 10
   fname += "dict_data_item_subj_v4_hidden_run4.pkl.p2.pbz2"     
elif dataset_id == "pubmed":
   num_batches = 1
   data_version_name = "2"
   fname += "pubmed_data_dict_v"+data_version_name+"_p2.pkl.pbz2" 
elif dataset_id == "pubmed_heuristic":
   num_batches = 10
   data_version_name = "2"
   fname += "pubmed_heuristic_data_dict_v"+data_version_name+"_p2.pkl.pbz2"    
elif dataset_id == "freebase":
   gpu_id = 3
   num_batches = 5 
   data_version_name = "5"
   fname += "freebase_data_dict_v"+data_version_name+"_p2.pkl.pbz2" 
elif dataset_id == "genephene":
   gpu_id = 2
   num_batches = 5
   data_version_name = "2"
   fname += "genephene_data_dict_v"+data_version_name+"_p2.pkl.pbz2"
else:
   assert False,"Unknown dataset_id: "+str(dataset_id) 

print("data_version_name: ",data_version_name)
#

out_dir = "./out_clust_single_pass/"+abl_category+"/"+dataset_id+"/"+"version_"+data_version_name+"/"
#
out_fname_perf = out_dir + "out_perf_"+dataset_id+".pkl"
out_fname_hypparams = out_dir + "out_best_hypparams_"+dataset_id+".pkl"
print("#")
print("fname: ",fname)
print("out_dir: ",out_dir)
print("#")
print("is_rand_batch_train: ",is_rand_batch_train)
print("#")
#
#
#is_gpu = True
if is_gpu:
    #gpu_id = 0
    torch.cuda.set_device(int(gpu_id))

# #common params 
# k = 50
# kf = None #0.000001
# num_layers = 1
# e_actf = "tanh"
# k_neigh = 100
# #
# learning_rate = 1e-4
# #learning_rate = 1e-4 #
# weight_decay = 1e-4
# convg_thres = -1e-3 #-1e-3
# max_epochs = 5000 #3000 #40000
# #
# is_pretrain = False #False
# learning_rate_pretrain = 1e-3
# weight_decay_pretrain = 0.001
# convg_thres_pretrain = None #1e-2
# max_epochs_pretrain = 2
# #
# mini_batch_size_frac = 1.0
# #
# is_train = True
# #is_train = False
# is_load_init = False
# #
# is_fac_pp = True

# #
# print("#")
# print("is_train: ",is_train)
# print("is_load_init: ",is_load_init)
# print("is_fac_pp: ",is_fac_pp)
# print("#")

#data_dict = pkl.load(open(fname,"rb"))
data_dict = decompress_pickle(fname)

if dataset_id in ["wiki1","wiki2","wiki3","wiki4"]:

    #common params 
    k = 100
    kf = None #0.000001
    num_layers = 2
    e_actf = "tanh"
    k_neigh = 100
    #
    learning_rate = 1e-4
    if dataset_id in ["wiki1"]:
        weight_decay = 1e-5
    else:
        weight_decay = 1e-4
    #weight_decay = 1e-4 #wiki1 1e-3, wiki2-4 1e-4
    convg_thres = -1e-3 #-1e-3
    max_epochs = 2500 
    #max_epochs = 50
    #
    is_pretrain = False #False
    learning_rate_pretrain = 1e-3
    weight_decay_pretrain = 0.001
    convg_thres_pretrain = None #1e-2
    max_epochs_pretrain = 2
    #
    mini_batch_size_frac = 1.0
    #
    is_train = True
    #is_train = False
    is_load_init = False
    #
    is_fac_pp = True

    #
    print("#")
    print("is_train: ",is_train)
    print("is_load_init: ",is_load_init)
    print("is_fac_pp: ",is_fac_pp)
    print("#")
    
    X_ts = data_dict["data_hidden"]['X_item_subj']
    X_sb = data_dict["data_hidden"]['X_subj_bow']
    X_tb = data_dict["data_hidden"]['X_item_bow']

    #
    print("#")
    print("Before pp:")
    print("#")
    print("X_ts:")
    print("min: ",X_ts.min()," max: ",X_ts.max()," mean: ",X_ts.mean()," sd: ",X_ts.std())
    print("X_sb:")
    print("min: ",X_sb.min()," max: ",X_sb.max()," mean: ",X_sb.mean()," sd: ",X_sb.std())
    print("X_tb:")
    print("min: ",X_tb.min()," max: ",X_tb.max()," mean: ",X_tb.mean()," sd: ",X_tb.std())

    X_data_bef_pp = {
        "X_ts":X_ts, \
        "X_sb":X_sb,
        "X_tb":X_tb}

    scaler1 = StandardScaler()
    X_ts = scaler1.fit_transform(X_ts)
    scaler2 = StandardScaler()
    X_sb = scaler2.fit_transform(X_sb)
    scaler3 = StandardScaler()
    X_tb = scaler3.fit_transform(X_tb)

    #_, X_sb = get_size_fac_pp(X_sb)
    #_, X_t2b = get_size_fac_pp(X_t2b)

    # In[16]:

    print("#")
    print("After pp:")
    print("#")
    print("X_ts:")
    print("min: ",X_ts.min()," max: ",X_ts.max()," mean: ",X_ts.mean()," sd: ",X_ts.std())
    print("X_sb:")
    print("min: ",X_sb.min()," max: ",X_sb.max()," mean: ",X_sb.mean()," sd: ",X_sb.std())
    print("X_tb:")
    print("min: ",X_tb.min()," max: ",X_tb.max()," mean: ",X_tb.mean()," sd: ",X_tb.std())

    y_t = data_dict["data_hidden"]["cluster_labels"]

    print("#")
    print("X_ts.shape: ",X_ts.shape)
    print("X_sb.shape: ",X_sb.shape)
    print("X_tb.shape: ",X_tb.shape)
    print("#")
    print("y_t.shape: ",len(y_t))
    print("#")


    dict_e_size = {}
    dict_e_size["t"] = X_ts.shape[0]
    dict_e_size["s"] = X_ts.shape[1]
    dict_e_size["b"] = X_sb.shape[1]


    X_ts = torch.from_numpy(X_ts).float()
    X_sb = torch.from_numpy(X_sb).float()
    X_tb = torch.from_numpy(X_tb).float()

    if is_gpu:
        X_ts = X_ts.cuda()
        X_sb = X_sb.cuda()
        X_tb = X_tb.cuda()

    dict_num_clusters =  {"t":3,"s":3,"b":3}

    dict_e_loss_weight = {
                            "t":1.0,\
                            "s":1.0,\
                            "b":1.0}
    dict_loss_weight = {
                            "aec":1.0,
                            "mat":1.0,
                            "clust":1.0
                        }
    

    G = {
        "t":["X_ts","X_tb"],\
        "s":["X_ts","X_sb"],\
        "b":["X_tb","X_sb"]}

    X_data = {
        "X_ts":X_ts, \
        "X_tb":X_tb, \
        "X_sb":X_sb
    }

    X_meta = {
        "X_ts":["t","s"],\
        "X_tb":["t","b"],\
        "X_sb":["s","b"]
    }

    X_dtype = {
        "X_ts":"real", \
        "X_tb":"real",\
        "X_sb":"real"
    }

    y_val_dict = {}
    y_val_dict["t"] = np.array(y_t)
    y_val_dict["b"] = np.array([])
    y_val_dict["s"] = np.array([])

elif dataset_id == "pubmed":

    #common params 
    k = 100
    kf = 0.000001
    num_layers = 2
    e_actf = "tanh"
    k_neigh = 100
    #
    #learning_rate = 1e-4
    learning_rate = 1e-6 #
    weight_decay = 1e-3
    convg_thres = -1e-3 #-1e-3
    max_epochs = 2000 
    #max_epochs = 20
    #
    is_pretrain = False #False
    learning_rate_pretrain = 1e-3
    weight_decay_pretrain = 0.001
    convg_thres_pretrain = None #1e-2
    max_epochs_pretrain = 2
    #
    mini_batch_size_frac = 1.0
    #
    is_train = True
    #is_train = False
    is_load_init = False
    #
    is_fac_pp = True

    #
    print("#")
    print("is_train: ",is_train)
    print("is_load_init: ",is_load_init)
    print("is_fac_pp: ",is_fac_pp)
    print("#")

    list_x_id = list(data_dict["matrices"].keys())
    list_e_id = list(data_dict["metadata"]["dict_e_size"].keys())
    #
    #list_x_id = data_dict["list_x_id"]
    #list_e_id = data_dict["list_e_id"]
    #
    print("Loading the matrices: ")
    dict_id_X = {}
    X_data_bef_pp = {}
    for x_id in list_x_id:
        dict_id_X[x_id] = data_dict["matrices"][x_id]
        X_data_bef_pp[x_id] = dict_id_X[x_id]
        print("x_id: ",x_id," X.shape: ",dict_id_X[x_id].shape)
    print("#")
    #
    print("#")
    print("Before pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",dict_id_X[temp_id].min()," max: ",dict_id_X[temp_id].max()," mean: ",np.round(dict_id_X[temp_id].mean(),4)," sd: ",np.round(dict_id_X[temp_id].std(),4))
    print("#")
    #
    print("Preprocessing the matrices: ")
    dict_id_X_pp = {}
    X_data_size_fac = {}
    for temp_id in list_x_id:
        if is_fac_pp:
            _, dict_id_X_pp[temp_id] = get_size_fac_pp(dict_id_X[temp_id])
        else:
            scaler1 = StandardScaler()
            dict_id_X_pp[temp_id] = scaler1.fit_transform(dict_id_X[temp_id])
        print("temp_id: ",temp_id," X.shape: ",dict_id_X_pp[temp_id].shape)        
    print("#")
    #
    print("#")
    print("After pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",np.round(dict_id_X_pp[temp_id].min(),4)," max: ",np.round(dict_id_X_pp[temp_id].max(),4)," mean: ",np.round(dict_id_X_pp[temp_id].mean(),4)," sd: ",np.round(dict_id_X_pp[temp_id].std(),4))
    print("#")
    #
    X_meta = data_dict["metadata"]["x_meta"] 
    #
    G = {}
    for e_id in list_e_id:
        temp_list = []
        for x_id in X_meta.keys():
            if e_id in X_meta[x_id]:
                temp_list.append(x_id)
        G[e_id] = temp_list
    #
    print("#")
    print("G: ")
    print("#")
    print(G)
    print("#")
    #
    y_val_dict = {}
    for e_id in G.keys():
        if e_id in ["e2"]:
            y_val_dict[e_id] = np.array(data_dict["gt"]["list_e2_labels"])
        else:
            y_val_dict[e_id] = np.array([])
    #
    X_data = {}
    for x_id in dict_id_X_pp:
        if is_gpu:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float().cuda()
        else:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float()
        print("X"+x_id,", shape: ",X_data[x_id].shape)
    #
    dict_e_size = {}
    for e_id in G.keys():
        x_id = G[e_id][0]
        if X_meta[x_id][0] == e_id:
            dict_e_size[e_id] = X_data[x_id].shape[0]
        else:
            dict_e_size[e_id] = X_data[x_id].shape[1]
    #
    print("#")
    print("dict_e_size: ")
    print("#")
    for e_id in dict_e_size.keys():
        print("e_id: ",e_id,", size: ",dict_e_size[e_id])
    print("#")
    #
    num_cluster = 8 #books has 8 classes #cluster all entities to have the same number of clusters
    dict_num_clusters = {}
    dict_e_loss_weight = {}
    for e_id in list_e_id:
        dict_num_clusters[e_id] = num_cluster
        dict_e_loss_weight[e_id] = 1.0
    #
    dict_loss_weight = {
                        "aec":1.0,
                        "mat":1.0,
                        "clust":1.0
                    }
    #
    X_dtype = {}
    for x_id in X_data:
        X_dtype[x_id] = "real" 

elif dataset_id == "pubmed_heuristic":

    #common params 
    k = 100
    kf = None #0.5
    num_layers = 2
    e_actf = "tanh"
    k_neigh = 100
    #
    #learning_rate = 1e-6
    learning_rate = 1e-4
    weight_decay = 1e-3
    convg_thres = -1e-3 #-1e-3
    max_epochs = 2000 #9000 #9000 for v2 of pubmed heuristic data
    #max_epochs = 50
    #
    is_pretrain = False #False
    learning_rate_pretrain = 1e-3
    weight_decay_pretrain = 0.001
    convg_thres_pretrain = None #1e-2
    max_epochs_pretrain = 2
    #
    mini_batch_size_frac = 1.0
    #
    is_train = True
    #is_train = False
    is_load_init = False
    #
    is_fac_pp = False

    #
    print("#")
    print("is_train: ",is_train)
    print("is_load_init: ",is_load_init)
    print("is_fac_pp: ",is_fac_pp)
    print("#")

    list_x_id = list(data_dict["matrices"].keys())
    list_e_id = list(data_dict["metadata"]["dict_e_size"].keys())
    #
    #list_x_id = data_dict["list_x_id"]
    #list_e_id = data_dict["list_e_id"]
    #
    print("Loading the matrices: ")
    dict_id_X = {}
    X_data_bef_pp = {}
    for x_id in list_x_id:
        dict_id_X[x_id] = data_dict["matrices"][x_id]
        X_data_bef_pp[x_id] = dict_id_X[x_id]
        print("x_id: ",x_id," X.shape: ",dict_id_X[x_id].shape)
    print("#")
    #
    print("#")
    print("Before pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",dict_id_X[temp_id].min()," max: ",dict_id_X[temp_id].max()," mean: ",np.round(dict_id_X[temp_id].mean(),4)," sd: ",np.round(dict_id_X[temp_id].std(),4))
    print("#")
    #
    print("Preprocessing the matrices: ")
    dict_id_X_pp = {}
    X_data_size_fac = {}
    for temp_id in list_x_id:
        if is_fac_pp:
            _, dict_id_X_pp[temp_id] = get_size_fac_pp(dict_id_X[temp_id])
        else:
            epsilon = 1e-10
            X_temp = dict_id_X[temp_id]
            # X_temp = pd.DataFrame(X_temp).fillna(0).to_numpy()
            # X_temp = np.log(X_temp + 1.0) + epsilon
            # X_temp = np.nan_to_num(X_temp)
            scaler1 = StandardScaler()
            dict_id_X_pp[temp_id] = scaler1.fit_transform(X_temp) #dict_id_X[temp_id])
        print("temp_id: ",temp_id," X.shape: ",dict_id_X_pp[temp_id].shape)        
    print("#")
    #
    print("#")
    print("After pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",np.round(dict_id_X_pp[temp_id].min(),4)," max: ",np.round(dict_id_X_pp[temp_id].max(),4)," mean: ",np.round(dict_id_X_pp[temp_id].mean(),4)," sd: ",np.round(dict_id_X_pp[temp_id].std(),4))
    print("#")
    #
    X_meta = data_dict["metadata"]["x_meta"] 
    #
    G = {}
    for e_id in list_e_id:
        temp_list = []
        for x_id in X_meta.keys():
            if e_id in X_meta[x_id]:
                temp_list.append(x_id)
        G[e_id] = temp_list
    #
    print("#")
    print("G: ")
    print("#")
    print(G)
    print("#")
    #
    y_val_dict = {}
    for e_id in G.keys():
        if e_id in ["e2"]:
            y_val_dict[e_id] = np.array(data_dict["gt"]["list_e2_labels"])
        else:
            y_val_dict[e_id] = np.array([])
    #
    X_data = {}
    for x_id in dict_id_X_pp:
        if is_gpu:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float().cuda()
        else:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float()
        print("X"+x_id,", shape: ",X_data[x_id].shape)
    #
    dict_e_size = {}
    for e_id in G.keys():
        x_id = G[e_id][0]
        if X_meta[x_id][0] == e_id:
            dict_e_size[e_id] = X_data[x_id].shape[0]
        else:
            dict_e_size[e_id] = X_data[x_id].shape[1]
    #
    print("#")
    print("dict_e_size: ")
    print("#")
    for e_id in dict_e_size.keys():
        print("e_id: ",e_id,", size: ",dict_e_size[e_id])
    print("#")
    #
    num_cluster = 8 #books has 8 classes #cluster all entities to have the same number of clusters
    dict_num_clusters = {}
    dict_e_loss_weight = {}
    for e_id in list_e_id:
        dict_num_clusters[e_id] = num_cluster
        dict_e_loss_weight[e_id] = 1.0
    #
    dict_loss_weight = {
                        "aec":1.0,
                        "mat":1.0,
                        "clust":1.0
                    }
    #
    X_dtype = {}
    for x_id in X_data:
        X_dtype[x_id] = "real"

elif dataset_id == "freebase":

    #common params 
    k = 100
    kf = None #0.0000001
    num_layers = 2
    e_actf = "tanh"
    k_neigh = 100
    #
    learning_rate = 1e-6
    #learning_rate = 1e-4 #
    weight_decay = 1e-3
    convg_thres = -1e-3 #-1e-3
    #max_epochs = 6000 #3000 #40000
    max_epochs = 2000
    #
    is_pretrain = False #False
    learning_rate_pretrain = 1e-3
    weight_decay_pretrain = 0.001
    convg_thres_pretrain = None #1e-2
    max_epochs_pretrain = 2
    #
    mini_batch_size_frac = 1.0
    #
    is_train = True
    #is_train = False
    is_load_init = False
    #
    is_fac_pp = True

    #
    print("#")
    print("is_train: ",is_train)
    print("is_load_init: ",is_load_init)
    print("is_fac_pp: ",is_fac_pp)
    print("#")

    #
    list_x_id = list(data_dict["matrices"].keys())
    list_e_id = list(data_dict["metadata"]["dict_e_size"].keys())
    #
    #list_x_id = data_dict["list_x_id"]
    #list_e_id = data_dict["list_e_id"]
    #
    print("Loading the matrices: ")
    dict_id_X = {}
    X_data_bef_pp = {}
    for x_id in list_x_id:
        dict_id_X[x_id] = data_dict["matrices"][x_id]
        X_data_bef_pp[x_id] = dict_id_X[x_id]
        print("x_id: ",x_id," X.shape: ",dict_id_X[x_id].shape)
    print("#")
    #
    print("#")
    print("Before pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",dict_id_X[temp_id].min()," max: ",dict_id_X[temp_id].max()," mean: ",np.round(dict_id_X[temp_id].mean(),4)," sd: ",np.round(dict_id_X[temp_id].std(),4))
    print("#")
    #
    print("Preprocessing the matrices: ")
    dict_id_X_pp = {}
    X_data_size_fac = {}
    for temp_id in list_x_id:
        if is_fac_pp:
            _, dict_id_X_pp[temp_id] = get_size_fac_pp(dict_id_X[temp_id])
        else:
            scaler1 = StandardScaler()
            dict_id_X_pp[temp_id] = scaler1.fit_transform(dict_id_X[temp_id])
        print("temp_id: ",temp_id," X.shape: ",dict_id_X_pp[temp_id].shape)        
    print("#")
    #
    print("#")
    print("After pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",np.round(dict_id_X_pp[temp_id].min(),4)," max: ",np.round(dict_id_X_pp[temp_id].max(),4)," mean: ",np.round(dict_id_X_pp[temp_id].mean(),4)," sd: ",np.round(dict_id_X_pp[temp_id].std(),4))
    print("#")
    #
    X_meta = data_dict["metadata"]["x_meta"] 
    #
    G = {}
    for e_id in list_e_id:
        temp_list = []
        for x_id in X_meta.keys():
            if e_id in X_meta[x_id]:
                temp_list.append(x_id)
        G[e_id] = temp_list
    #
    print("#")
    print("G: ")
    print("#")
    print(G)
    print("#")
    #
    y_val_dict = {}
    for e_id in G.keys():
        if e_id in ["e1"]:
            y_val_dict[e_id] = np.array(data_dict["gt"]["list_e1_labels"])
        else:
            y_val_dict[e_id] = np.array([])
    #
    X_data = {}
    for x_id in dict_id_X_pp:
        if is_gpu:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float().cuda()
        else:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float()
        print("X"+x_id,", shape: ",X_data[x_id].shape)
    #
    dict_e_size = {}
    for e_id in G.keys():
        x_id = G[e_id][0]
        if X_meta[x_id][0] == e_id:
            dict_e_size[e_id] = X_data[x_id].shape[0]
        else:
            dict_e_size[e_id] = X_data[x_id].shape[1]
    #
    print("#")
    print("dict_e_size: ")
    print("#")
    for e_id in dict_e_size.keys():
        print("e_id: ",e_id,", size: ",dict_e_size[e_id])
    print("#")
    #
    num_cluster = 8 #books has 8 classes #cluster all entities to have the same number of clusters
    dict_num_clusters = {}
    dict_e_loss_weight = {}
    for e_id in list_e_id:
        dict_num_clusters[e_id] = num_cluster
        dict_e_loss_weight[e_id] = 1.0
    #
    dict_loss_weight = {
                        "aec":1.0,
                        "mat":1.0,
                        "clust":1.0
                    }
    #
    X_dtype = {}
    for x_id in X_data:
        X_dtype[x_id] = "real"    

elif dataset_id == "genephene":
    #common params 
    k = 100
    kf = None #0.000001
    num_layers = 1
    e_actf = "tanh"
    k_neigh = 100
    #
    learning_rate = 1e-5
    #learning_rate = 1e-5
    weight_decay = 1e-3
    convg_thres = -1e-3 #-1e-3
    #max_epochs = 5000 #3000 #3000
    max_epochs = 1000
    #
    is_pretrain = False #False
    learning_rate_pretrain = 1e-3
    weight_decay_pretrain = 0.001
    convg_thres_pretrain = None #1e-2
    max_epochs_pretrain = 2
    #
    mini_batch_size_frac = 1.0
    #
    is_train = True
    #is_train = False
    is_load_init = False
    #
    is_fac_pp = False

    #
    print("#")
    print("is_train: ",is_train)
    print("is_load_init: ",is_load_init)
    print("is_fac_pp: ",is_fac_pp)
    print("#")    

    list_x_id = list(data_dict["matrices"].keys())
    list_e_id = list(data_dict["metadata"]["dict_e_size"].keys())
    #
    #list_x_id = data_dict["list_x_id"]
    #list_e_id = data_dict["list_e_id"]
    #
    print("Loading the matrices: ")
    dict_id_X = {}
    X_data_bef_pp = {}
    for x_id in list_x_id:
        dict_id_X[x_id] = data_dict["matrices"][x_id]
        X_data_bef_pp[x_id] = dict_id_X[x_id]
        print("x_id: ",x_id," X.shape: ",dict_id_X[x_id].shape)
    print("#")
    #
    print("#")
    print("Before pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",dict_id_X[temp_id].min()," max: ",dict_id_X[temp_id].max()," mean: ",np.round(dict_id_X[temp_id].mean(),4)," sd: ",np.round(dict_id_X[temp_id].std(),4))
    print("#")
    #
    print("Preprocessing the matrices: ")
    dict_id_X_pp = {}
    X_data_size_fac = {}
    epsilon = 1e-9
    for temp_id in list_x_id:
        print("temp_id: ",temp_id)
        # if is_fac_pp:
        #     _, dict_id_X_pp[temp_id] = get_size_fac_pp(dict_id_X[temp_id])
        # else:
        scaler1 = StandardScaler()
        #1
        #dict_id_X_pp[temp_id] = scaler1.fit_transform(dict_id_X[temp_id])
        #2
        #X_temp = np.log(np.nan_to_num(dict_id_X[temp_id]) + 1.0) + epsilon
        #3
        X_temp = dict_id_X[temp_id]
        X_temp = pd.DataFrame(X_temp).fillna(0).to_numpy()
        X_temp = np.log(X_temp + 1.0) + epsilon
        X_temp = np.nan_to_num(X_temp)
        #4
        #X_temp[np.isnan(X_temp)] = 0
        #5
        #X_temp = np.where(np.isnan(X_temp), 0, X_temp)
        print("X_temp: ")
        print(X_temp)
        dict_id_X_pp[temp_id] = scaler1.fit_transform(X_temp)
        print("temp_id: ",temp_id," X.shape: ",dict_id_X_pp[temp_id].shape)        
    print("#")
    #
    print("#")
    print("After pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",np.round(dict_id_X_pp[temp_id].min(),4)," max: ",np.round(dict_id_X_pp[temp_id].max(),4)," mean: ",np.round(dict_id_X_pp[temp_id].mean(),4)," sd: ",np.round(dict_id_X_pp[temp_id].std(),4))
    print("#")
    #
    X_meta = data_dict["metadata"]["x_meta"] 
    #
    G = {}
    for e_id in list_e_id:
        temp_list = []
        for x_id in X_meta.keys():
            if e_id in X_meta[x_id]:
                temp_list.append(x_id)
        G[e_id] = temp_list
    #
    print("#")
    print("G: ")
    print("#")
    print(G)
    print("#")
    #
    y_val_dict = {}
    for e_id in G.keys():
        if e_id in ["p"]:
            if data_version_name in ["1"]:
                y_val_dict[e_id] = np.array(data_dict["gt"]["list_p_vitality_status_labels"])
            elif data_version_name in ["2"]:
                y_val_dict[e_id] = np.array(data_dict["gt"]["list_p_cancer_stage_labels"])
            else:
                assert False,"Unknown data_version_name: "+str(data_version_name)+" for dataset_id: "+str(dataset_id)
        else:
            y_val_dict[e_id] = np.array([])
    #
    X_data = {}
    for x_id in dict_id_X_pp:
        if is_gpu:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float().cuda()
        else:
            X_data[x_id] = torch.from_numpy(dict_id_X_pp[x_id]).float()
        print("X"+x_id,", shape: ",X_data[x_id].shape)
    #
    dict_e_size = {}
    for e_id in G.keys():
        x_id = G[e_id][0]
        if X_meta[x_id][0] == e_id:
            dict_e_size[e_id] = X_data[x_id].shape[0]
        else:
            dict_e_size[e_id] = X_data[x_id].shape[1]
    #
    print("#")
    print("dict_e_size: ")
    print("#")
    for e_id in dict_e_size.keys():
        print("e_id: ",e_id,", size: ",dict_e_size[e_id])
    print("#")
    #
    if data_version_name in ["1"]:
        num_cluster = 2 
    elif data_version_name in ["2"]:
        num_cluster = 4
    else:
        assert False,"Unknown data_version_name: "+str(data_version_name)+" for dataset_id: "+str(dataset_id)
    
    dict_num_clusters = {}
    dict_e_loss_weight = {}
    for e_id in list_e_id:
        dict_num_clusters[e_id] = num_cluster
        dict_e_loss_weight[e_id] = 1.0
    #
    dict_loss_weight = {
                        "aec":1.0,
                        "mat":1.0,
                        "clust":1.0
                    }
    #
    X_dtype = {}
    for x_id in X_data:
        X_dtype[x_id] = "real" 

else:
    assert False

def run_dcmtf(parameterization):
    #learning_rate = parameterization["learning_rate"]
    #weight_decay = parameterization["weight_decay"]
    #e_actf = parameterization["e_actf"]
    #num_layers = parameterization["num_layers"]
    #is_pretrain = parameterization["is_pretrain"]
    #k = parameterization["k"]
    #
    dcmtf_instance = dcmtf.dcmtf(G, X_data, X_data_bef_pp, X_meta, X_dtype,\
            k, kf, num_layers, e_actf, dict_num_clusters,\
            learning_rate, weight_decay, convg_thres, max_epochs,\
            is_pretrain, learning_rate_pretrain, weight_decay_pretrain, convg_thres_pretrain, max_epochs_pretrain,\
            mini_batch_size_frac, dict_e_loss_weight, dict_loss_weight,\
            dict_e_size, y_val_dict,\
            is_gpu, is_train, is_load_init, model_dir=out_dir)
    #
    dcmtf_instance.fit()
    #
    dict_ari_u = dcmtf_instance.calc_kmeans_ari_u()
    dict_ari_c = dcmtf_instance.calc_kmeans_ari_c()
    #batchmix_entropy = dcmtf_instance.calc_batchmix_entropy(list(y_val_dict.keys()),k_neigh)
    #align_score = dcmtf_instance.calc_align_score(list(y_val_dict.keys()),k_neigh)
    #dict_agree_score = dcmtf_instance.calc_agree_score(list(y_val_dict.keys()),k_neigh)

    out_dict = {}
    #out_dict["batchmix_entropy"] = (batchmix_entropy,0.0)
    #out_dict["align_score"] = (align_score,0.0)
    for e_id in y_val_dict.keys():
        out_dict["ari_u_"+e_id] = (dict_ari_u[e_id],0.0)
        out_dict["ari_c_"+e_id] = (dict_ari_c[e_id],0.0)
        #out_dict["ags_"+e_id] = (dict_agree_score[e_id],0.0)
    #    
    out_dict["loss"] = (dcmtf_instance.loss, 0.0)
    print("###")
    print("out_dict")
    print("###")
    pp.pprint(out_dict)
    print("###")
    return out_dict


dict_setting_perf = {}
dict_setting_besthyperparams = {}

print("#")
print("###########")
print("dataset_id: ",dataset_id)
print("###########")
print("#")
#
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

dict_batch_size = {}
dict_mini_batch_size = {}
for e_id in G.keys():
    dict_batch_size[e_id] = dict_e_size[e_id]
    dict_mini_batch_size[e_id] = int(dict_batch_size[e_id] * mini_batch_size_frac)

print("#")
print("G: ",G)
print("X_data: ",X_data)
print("X_meta: ",X_meta)
print("X_dtype: ",X_dtype)
print("#")
print("dict_e_size: ")
print(dict_e_size)
print("#")
print("dict_mini_batch_size: ")
print(dict_mini_batch_size)
print("#")


    # best_parameters, values, experiment, model = optimize(
    #     parameters=[
    #         {
    #             "name": "weight_decay",
    #             "type": "range",
    #             "bounds": [1e-6, 1e-2],
    #             "value_type": "float",  # Optional, defaults to inference from type of "bounds".
    #             "log_scale": False,  # Optional, defaults to False.
    #         },
    #         {
    #             "name": "learning_rate",
    #             "type": "range",
    #             "bounds": [1e-6, 1e-4],
    #             "value_type": "float",  # Optional, defaults to inference from type of "bounds".
    #             "log_scale": False,  # Optional, defaults to False.
    #         },
    #         # {
    #         #     "name": "k",
    #         #     "type": "choice",
    #         #     "values": [50, 100, 150, 200],
    #         #     "value_type": "int"
    #         # },
    #         # {
    #         #     "name": "e_actf",
    #         #     "type": "choice",
    #         #     "values": ["tanh", "lrelu"],
    #         #     "value_type": "str"
    #         # },
    #         # {
    #         #     "name": "num_layers",
    #         #     "type": "choice",
    #         #     "values": [1,2],
    #         #     "value_type": "int"
    #         # }
    #     ],
    #     experiment_name="dcmtf_bo",
    #     objective_name="loss",
    #     evaluation_function=run_dcmtf,
    #     minimize=True,  # Optional, defaults to False.
    #     #parameter_constraints=["k%2 <= 0"],  # Optional.
    #     #outcome_constraints=["loss >= 0"],  # Optional.
    #     total_trials=2, # Optional.
    # )


    # # In[29]:

    # print("experiment.trials: ")
    # print(experiment.trials)
    # print("#")

    # # In[30]:

    # print("best_parameters: ")
    # print(best_parameters)
    # print("#")

    # # In[31]:

    # print("values[0]: ")
    # print(values[0])
    # print("#")

    # # In[32]:


    # for idx in experiment.trials.keys():
    #     trial =  experiment.trials[idx]
    #     print("obj: ",round(trial.objective_mean,4)," params: ",trial.arm.parameters)
    # print("#")

    # In[33]:


    # # `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
    # # optimization runs, so we wrap out best objectives array in another array.
    # best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
    # best_objective_plot = optimization_trace_single_method(
    #     y=np.minimum.accumulate(best_objectives, axis=1),
    #     #optimum=hartmann6.fmin,
    #     title="Model performance vs. # of iterations",
    #     ylabel="loss",
    # )
    # render(best_objective_plot)


    # # In[34]:


    # render(plot_contour(model=model, param_x='learning_rate', param_y='weight_decay', metric_name='loss'))


    # # In[35]:


    # render(plot_contour(model=model, param_x='learning_rate', param_y='weight_decay', metric_name='c1'))


    # # In[ ]:


    # render(plot_contour(model=model, param_x='learning_rate', param_y='weight_decay', metric_name='c2'))


    # # In[ ]:




    #print("####")
    #print("Re-run with best parameters: ")

    #learning_rate = best_parameters["learning_rate"]
    #weight_decay = best_parameters["weight_decay"]
    #e_actf = parameterization["e_actf"]
    #num_layers = parameterization["num_layers"]
    #is_pretrain = parameterization["is_pretrain"]
    #k = parameterization["k"]
    #
# dcmtf_instance = dcmtf(G, X_data, X_data_bef_pp, X_meta, X_dtype,\
#         k, kf, num_layers, e_actf, dict_num_clusters,\
#         learning_rate, weight_decay, convg_thres, max_epochs,\
#         is_pretrain, learning_rate_pretrain, weight_decay_pretrain, convg_thres_pretrain, max_epochs_pretrain,\
#         mini_batch_size_frac, num_batches, dict_e_loss_weight, dict_loss_weight,\
#         dict_e_size, y_val_dict,\
#         is_gpu, is_train, is_load_init, model_dir=out_dir)

#
#dcmtf_instance.fit()

if dataset_id in ["pubmed"]:
    num_runs = 3 #hyperparameter
    print("#####")
    print("dataset_id: ",dataset_id)
    print("###")
    print("num_runs: ",num_runs)
    print("###")
    for i in np.arange(num_runs):
        if i+1 > 1 :
            is_train = False
        print("###")
        print("Cur pubmed run#: ",i+1)
        print("is_train: ",is_train)
        print("###")
        #
        dcmtf_instance = dcmtfsinglepass(G, X_data, X_data_bef_pp, X_data_size_fac, X_meta, X_dtype,\
                k, kf, num_layers, e_actf, dict_num_clusters,\
                learning_rate, weight_decay, convg_thres, max_epochs,\
                is_pretrain, learning_rate_pretrain, weight_decay_pretrain, convg_thres_pretrain, max_epochs_pretrain,\
                mini_batch_size_frac, num_batches, dict_e_loss_weight, dict_loss_weight,\
                dict_e_size, y_val_dict,\
                is_gpu, is_train, is_load_init, is_rand_batch_train, \
                model_dir=out_dir, abl_category=abl_category, is_single_pass=is_single_pass)
        #
        dcmtf_instance.fit()
        #
        dcmtf_instance.persist_out(out_dir)
        print("#")
        print("Calculating model performance: ")
        print("#")
        dict_ari_u = dcmtf_instance.calc_kmeans_ari_u()
        dict_ari_c = dcmtf_instance.calc_kmeans_ari_c()
        print("###")
elif dataset_id in ["pubmed_heuristic"]:
    num_runs = 2 #hyperparameter
    print("#####")
    print("dataset_id: ",dataset_id)
    print("###")
    print("num_runs: ",num_runs)
    print("###")
    for i in np.arange(num_runs):
        if i+1 > 1 :
            is_train = False
            learning_rate = 1e-5
        print("###")
        print("Cur pubmed DA run#: ",i+1)
        print("is_train: ",is_train)
        print("###")
        #
        dcmtf_instance = dcmtfsinglepass(G, X_data, X_data_bef_pp, X_data_size_fac, X_meta, X_dtype,\
                k, kf, num_layers, e_actf, dict_num_clusters,\
                learning_rate, weight_decay, convg_thres, max_epochs,\
                is_pretrain, learning_rate_pretrain, weight_decay_pretrain, convg_thres_pretrain, max_epochs_pretrain,\
                mini_batch_size_frac, num_batches, dict_e_loss_weight, dict_loss_weight,\
                dict_e_size, y_val_dict,\
                is_gpu, is_train, is_load_init, is_rand_batch_train, \
                model_dir=out_dir, abl_category=abl_category, is_single_pass=is_single_pass)
        #
        dcmtf_instance.fit()
        #
        dcmtf_instance.persist_out(out_dir)
        print("#")
        print("Calculating model performance: ")
        print("#")
        dict_ari_u = dcmtf_instance.calc_kmeans_ari_u()
        dict_ari_c = dcmtf_instance.calc_kmeans_ari_c()
        print("###")
else:
    dcmtf_instance = dcmtfsinglepass(G, X_data, X_data_bef_pp, X_data_size_fac, X_meta, X_dtype,\
            k, kf, num_layers, e_actf, dict_num_clusters,\
            learning_rate, weight_decay, convg_thres, max_epochs,\
            is_pretrain, learning_rate_pretrain, weight_decay_pretrain, convg_thres_pretrain, max_epochs_pretrain,\
            mini_batch_size_frac, num_batches, dict_e_loss_weight, dict_loss_weight,\
            dict_e_size, y_val_dict,\
            is_gpu, is_train, is_load_init, is_rand_batch_train, \
            model_dir=out_dir, abl_category=abl_category, is_single_pass=is_single_pass)
    #
    dcmtf_instance.fit()
    #
    dcmtf_instance.persist_out(out_dir)

# #
# print("#")
# print("Persisting model and outputs: ")
# print("#")
# fname = out_dir+"dict_vae.pkl"
# print("Persisting: ",fname)
# pkl.dump(dcmtf_instance.dict_vae,open(fname,"wb"))

# fname = out_dir+"dict_ffnu_cat.pkl"
# print("Persisting: ",fname)
# pkl.dump(dcmtf_instance.dict_ffnu_cat,open(fname,"wb"))

# fname = out_dir+"dict_ffn_clust.pkl"
# print("Persisting: ",fname)
# pkl.dump(dcmtf_instance.dict_ffn_clust,open(fname,"wb"))

# fname = out_dir+"dict_A.pkl"
# print("Persisting: ",fname)
# dict_temp = dcmtf_instance.dict_A
# dict_temp_np = {}
# for temp_key in dict_temp.keys():
#     dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
# pkl.dump(dict_temp_np,open(fname,"wb"))

# fname = out_dir+"dict_u_clust_labels.pkl"
# print("Persisting: ",fname)
# dict_u_clust_labels = dcmtf_instance.get_u_clust_labels()
# pkl.dump(dict_u_clust_labels,open(fname,"wb"))

# fname = out_dir+"dict_c_clust_labels.pkl"
# print("Persisting: ",fname)
# dict_c_clust_labels = dcmtf_instance.get_c_clust_labels()
# pkl.dump(dict_c_clust_labels,open(fname,"wb"))

# fname = out_dir+"dict_recons_X.pkl"
# print("Persisting: ",fname)
# dict_temp = dcmtf_instance.dict_recons_X
# dict_temp_np = {}
# for temp_key in dict_temp.keys():
#     dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
# pkl.dump(dict_temp_np,open(fname,"wb"))

# fname = out_dir+"dict_recons_Y.pkl"
# print("Persisting: ",fname)
# dict_temp = dcmtf_instance.dict_C_dec
# dict_temp_np = {}
# for temp_key in dict_temp.keys():
#     dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
# pkl.dump(dict_temp_np,open(fname,"wb"))

# fname = out_dir+"dict_I_ortho.pkl"
# print("Persisting: ",fname)
# dict_temp = dcmtf_instance.dict_I_ortho
# dict_temp_np = {}
# for temp_key in dict_temp.keys():
#     dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
# pkl.dump(dict_temp_np,open(fname,"wb"))

# fname = out_dir+"dict_U.pkl"
# print("Persisting: ",fname)
# dict_temp = dcmtf_instance.dict_U
# dict_temp_np = {}
# for temp_key in dict_temp.keys():
#     dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
# pkl.dump(dict_temp_np,open(fname,"wb"))

# fname = out_dir+"dict_mu.pkl"
# print("Persisting: ",fname)
# dict_temp = dcmtf_instance.dict_mu
# dict_temp_np = {}
# for temp_key in dict_temp.keys():
#     dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
# pkl.dump(dict_temp_np,open(fname,"wb"))

# fname = out_dir+"dict_out_params.pkl"
# print("Persisting: ",fname)
# pkl.dump(dcmtf_instance.out_params,open(fname,"wb"))
# print("#")
#
print("#")
print("Out params: ")
print("#")
pp.pprint(dcmtf_instance.out_params)
#
print("#")
print("Calculating model performance: ")
print("###")
dict_ari_u = dcmtf_instance.calc_kmeans_ari_u()
dict_ari_c = dcmtf_instance.calc_kmeans_ari_c()
print("###")
dict_sil_u = dcmtf_instance.calc_kmeans_sil_score_u()
dict_sil_c = dcmtf_instance.calc_kmeans_sil_score_c()
#batchmix_entropy = dcmtf_instance.calc_batchmix_entropy(list(y_val_dict.keys()),k_neigh)
#align_score = dcmtf_instance.calc_align_score(list(y_val_dict.keys()),k_neigh)
#dict_agree_score = dcmtf_instance.calc_agree_score(list(y_val_dict.keys()),k_neigh)

# out_dict = {}
# #out_dict["batchmix_entropy"] = (batchmix_entropy,0.0)
# #out_dict["align_score"] = (align_score,0.0)
# for e_id in y_val_dict.keys():
#     out_dict["ari_u_"+e_id] = (dict_ari_u[e_id],0.0)
#     out_dict["ari_c_"+e_id] = (dict_ari_c[e_id],0.0)
#     # if e_id == "g":
#     #     out_dict["ags_"+e_id] = (-1,0.0)
#     # else:
#     #out_dict["ags_"+e_id] = (dict_agree_score[e_id],0.0)
# #    
# out_dict["loss"] = (dcmtf_instance.loss, 0.0)
# print("###")
# print("out_dict")
# print("###")
# pp.pprint(out_dict)
# print("###")
# dict_setting_perf = out_dict
    
# print("#")
# print("Experiment ended.")


# print("#")
# print("Persisting results: ")
# print("#")
# fname = out_dir+"dict_perf.pkl"
# print("Persisting: ",fname)
# pkl.dump(dict_setting_perf,open(fname,"wb"))
# print("###")

# # print("#")
# # print("dict_setting_perf")
# # print("###")
# # pp.pprint(dict_setting_perf)
# # print("#")
# # #
# # pkl.dump(dict_setting_perf,open(out_fname_perf,"wb"))
# #
# # for temp1 in dict_setting_perf.keys():
# #     for temp2 in dict_setting_perf[temp1].keys():
# #         dict_setting_perf[temp1][temp2] = np.round(dict_setting_perf[temp1][temp2][0],4)
# #
# print("###########################")        
# print("#")
# print("dict_perf")
# print("###")
# pp.pprint(pd.DataFrame(dict_setting_perf).T)
# print("#")

