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
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../mtl_lss/')
from Clustering_method import clustering


# %%
def estimate(x):
    l = len(x)-1
    mean = np.mean(x)
    std = np.sqrt(l*np.mean(abs(x - mean)**2))
    return mean, std


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
    epochs =50, 
    min_sep= min_sep, 
    max_sep= max_sep, 
    nedges= nedges, 
    lr=1e-4, 
    batch_size = 500, 
    pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', 
    pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy', 
    model_clustering='/nfs/pic.es/user/l/lcabayol/EUCLID/Improving-photometric-redshifts-using-large-scale-structure-and-neural-networks/models/2PCF2_600epochs.pt'
)

# %% jupyter={"outputs_hidden": true}
#Train the model with the data provided
Model.train_clustering(
    Nobj=1_000_000    
)

# %%
thetac = [0.00081062, 0.00213064, 0.00560019, 0.0147196 , 0.03868915,
       0.10169096, 0.26728557]

# %%
#Make prediction with new data
jackknife_preds = Model.pred_clustering(
    #Input the array of theta values to see the output of the predicted 2PCF
    theta_test=thetac
)


# %% jupyter={"outputs_hidden": true}
# np.delete?

# %%
def estimate(x):
    mean_x = np.mean(x,0)
    
    means = []
    for ii in range(len(x)):
        means.append(np.mean(np.delete(x,ii,axis=0),0))
        
    means = np.array(means)

    err = np.sqrt((len(x)-1)*np.sum((means - mean_x)**2,0))
    return mean_x, err


# %%
mean_nn, std_nn = estimate(jackknife_preds)

# %%

#plt.plot(thetac, mean, 'ro', markersize=4, color = 'navy',label = 'Counting pairs')
#plt.plot(thetac, jackknifes.mean(0), ls = '--', color = 'navy', markersize=4)
#plt.errorbar(thetac, mean, yerr=std, color='navy', fmt='o')

plt.plot(thetac, mean_nn, 'ro', markersize=4, color = 'crimson',label = 'Neural network')
plt.plot(thetac, jackknife_preds.mean(0), ls = '--', markersize=4, color = 'black')
plt.errorbar(thetac,mean_nn,std_nn, color = 'crimson', fmt='o')


plt.xscale('log')
plt.ylabel(r'$w(\theta)$', fontsize=16)
plt.xlabel(r'$\theta$', fontsize=16)
plt.grid(which='both')

plt.legend(fontsize = 12)

plt.savefig('wtheta.pdf')
plt.show()

# %%
