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

# %%
#Import clustering class from clustering_method script
sys.path.append('Clustering_method.py')
from clustering_method import clustering

# %%
#Initialize the class providing required and optional arguments
Model = clustering(
    
    #Required arguments (to pick by user)
    cluster_hlayers = 5, 
    epochs =2, 
    
    #Set by default arguments (if left blank)
    min_sep= 0.03, 
    max_sep= 26, 
    nedges= 8, 
    lr=1e-5 , 
    batch_size = 500, 
    pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', 
    pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy' 
)

# %%
#Train the model with the data provided
Model.train_clustering(
    Nobj=100    
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
