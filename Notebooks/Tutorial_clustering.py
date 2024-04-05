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
#     display_name: insight
#     language: python
#     name: insight
# ---

# %% [markdown]
# # TUTORIAL TO RUN A NETWORK PREDICTING THE 2p CORRELATION FUCNTION

# %% [markdown]
# This netowrk predicts the two point ocrrelation function of glaaxies from two files of angular distances. One of the files contains distances computed from galaxy positions from a simulated catalog, and the other file contains distances computed from a random catalog.

# %%
#Import clustering class from clustering_method script
import sys
sys.path.append('../mtl_lss/')
from Clustering_method import clustering

# %% [markdown]
# ## Initialize the method.

# %% [markdown]
# We first define the minimum and maximum angular separation considered. These values are in arcmin.

# %%
max_sep= 26
min_sep= 0.03
nedges= 8

# %%
#Initialize the class providing required and optional arguments
Model = clustering(
    cluster_hlayers = 5, 
    epochs =2, 
    min_sep= min_sep, 
    max_sep= max_sep, 
    nedges= nedges, 
    lr=1e-5, 
    batch_size = 500, 
    pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', 
    pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy' 
)

# %%
#Train the model with the data provided
Model.train_clustering(
    Nobj=100_000    
)

# %%
#Make prediction with new data
Model.pred_clustering(
    #Input the array of theta values to see the output of the predicted 2PCF
    theta_test=[0.00081062, 0.00213064, 0.00560019, 0.0147196 , 0.03868915,
       0.10169096, 0.26728557]
)

# %%
#Plot, waiting for fixed plot and training time decrease
