# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: env_ai
#     language: python
#     name: env_ai
# ---

# %% [markdown]
# # TUTORIAL TO RUN A NETWORK PREDICTING THE 2p CORRELATION FUCNTION

# %% [markdown]
# This netowrk predicts the two point ocrrelation function of glaaxies from two files of angular distances. One of the files contains distances computed from galaxy positions from a simulated catalog, and the other file contains distances computed from a random catalog.

# %%
#Import clustering class from clustering_method script
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../mtl_lss/')
from Clustering_method import clustering

# %% [markdown]
# We first define the minimum and maximum angular separation considered. These values are in arcmin.

# %%
max_sep= 26
min_sep= 0.03
nedges= 8

# %%
#Initialize the class providing required and optional arguments
Model = clustering(
    pathfile_distances,
    pathfile_drand,
    cluster_hlayers = 5, 
    epochs =50, 
    min_sep= min_sep, 
    max_sep= max_sep, 
    nedges= nedges, 
    lr=1e-4, 
    batch_size = 500, 
    model_clustering = None
)

# %% jupyter={"outputs_hidden": true}
#Train the model with the data provided
Model.train_clustering(
    Nobj='all' 
)

# %%
thetac = [0.00081062, 0.00213064, 0.00560019, 0.0147196 , 0.03868915,
       0.10169096, 0.26728557]

# %%
#Make prediction with new data
jackknife_preds = Model.pred_clustering(
    theta_test=thetac,
    plot = True
)
