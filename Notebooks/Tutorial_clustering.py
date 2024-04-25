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

# %%
#Import clustering class from clustering_method script
import sys
sys.path.append('architecture.py')
sys.path.append('script.py')
from script import clustering

# %%
#Initialize the class providing required and optional arguments
Model = clustering(
    
    #Required arguments (to pick by user)
    pathfile_distances,
    pathfile_drand,
    cluster_hlayers = 5, 
    epochs =2, 
    
    #Set by default arguments (if left blank)
    min_sep= 0.03, 
    max_sep= 26, 
    nedges= 8, 
    lr=1e-5 , 
    batch_size = 500
)

# %%
#Train the model with the data provided
Model.train_clustering(
    Nobj='all' 
)

# %%
#Make prediction with new data
Model.pred_clustering(
    #Input the array of theta values to see the output of the predicted 2PCF
    theta_test=[0.00081062, 0.00213064, 0.00560019, 0.0147196 , 0.03868915,
       0.10169096, 0.26728557],
    plot=True
)

# %%
