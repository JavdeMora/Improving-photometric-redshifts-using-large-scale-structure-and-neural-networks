#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Import clustering class from clustering_method script
sys.path.append('Clustering_method.py')
from clustering_method import clustering


# In[10]:


#Initialize the class providing required and optional arguments
Model = clustering(
    
    #Required arguments (to pick by user)
    cluster_hlayers = 5, #Number of hidden layers of the network
    epochs =2, #Number of epochs for the training
    
    #Set by default arguments (if left blank)
    min_sep= 0.03, #Minimum separation of galaxies
    max_sep= 26, #Maximum separation of galaxies
    nedges= 8, #Number of separation edges
    lr=1e-5 , #Learning rate for network training
    batch_size = 500, #Batch size for network training
    pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', #Path to file with real distances data
    pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy' #Path to file with random distances data
    
)


# In[11]:


#Train the model with the data provided
Model.train_clustering(
    
    #Set by default argument (if left blank)
    Nobj=100 #Number of rows of the training data subset
    
)


# In[12]:


#Make prediction with new data
Model.pred_clustering(
    #Input the array of theta values to see the output of the predicted 2PCF
    theta_test=[0.00081062, 0.00213064, 0.00560019, 0.0147196 , 0.03868915,
       0.10169096, 0.26728557]
)


# In[ ]:


#Plot, waiting for fixed plot and training time lowering

