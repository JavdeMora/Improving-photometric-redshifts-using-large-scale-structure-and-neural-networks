import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.cm as cm
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset
import sys
import scipy.stats as stats

sys.path.append('clustering_architecture.py')
from clustering_architecture import network_dists

class clustering:
    def __init__(self, cluster_hlayers, epochs, lr=1e-5 ,batch_size = 100, pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy'):
        self.net_2pcf= network_dists(cluster_hlayers).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        self.distances_array=self._get_distances_array()
    
    def _get_distances_array(self, pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy'):
        d = np.load(pathfile_distances)#[:,:100] <-- if you want to test #(400, 100000)
        drand = np.load(pathfile_drand)#[:,:100] <-- if you want to test
    
        distA = d.flatten()
        distB = drand.flatten()
    
        min_sep= 0.03
        max_sep = 26
        nedges=8
    
        th = np.linspace(np.log10(min_sep), np.log10(max_sep), nedges)
        theta = 10**th * (1./60.)
    
        Ndd = np.array([len(distA[(distA>theta[k])&(distA<theta[k+1])]) for k in range(len(theta)-1)])
        Ndd = np.append(Ndd,len(distA[(distA>theta[-1])]))
        Pdd = Ndd / Ndd.sum()
        wdd = Pdd.max() / Pdd 
    
        ranges =  [(theta[k],theta[k+1]) for k in range(len(theta)-1)]
        ranges.append((theta[-1],1.1))
        ranges = np.array(ranges)
        weights = wdd
    
        arr1 = d.copy().T
        arr2 = drand.copy().T
    
        # Labeling arrays
        labeled_arr1 = np.column_stack((arr1, np.zeros(arr1.shape[0], dtype=int)))  # Appending a column of 0s
        labeled_arr2 = np.column_stack((arr2, np.ones(arr2.shape[0], dtype=int)))  # Appending a column of 1s
    
        # Concatenating the labeled arrays vertically
        combined_array = np.vstack((labeled_arr1, labeled_arr2))
        # Creating the final list
        result_list = []
        for i in range(0,combined_array.shape[1]-1):
            for j in range(combined_array.shape[0]):
                value = combined_array[j, i]
                array_type = int(combined_array[j, -1])  # Extracting the array type (0 or 1)
                column_index = i  # Extracting the jk index (0 to 400)
    
                result_list.append([value, array_type, column_index])
    
        # Converting the list to a NumPy array
        distances_array = np.array(result_list)
    
        range_idx = np.searchsorted(ranges[:, 1], distances_array[:,0], side='right')
    
        w = weights[range_idx]
        distances_array = np.c_[distances_array, w.reshape(len(w),1)]
        distances_array = torch.Tensor(distances_array)
        
        return distances_array
    
    def train_clustering(self, epochs=2, Nobj=10, batch_size= 500, pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy',*args):
        distances_array=self.distances_array

        clustnet= self.net_2pcf

        optimizer = optim.Adam(clustnet.parameters(), lr=self.lr)# deberia separar entre lr photoz y lr clustering
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1200, gamma=0.01)
        CELoss = nn.CrossEntropyLoss(reduction='none')
        Nobj = 10
        for epoch in range(epochs):#deberia separar entre epochs photoz y epochs clustering
            print('starting epoch', epoch)

            distances_array_sub = distances_array[np.random.randint(0, distances_array.shape[0], distances_array.shape[0])]#revisar la size (yo he usado todas las distancias para entrenar, preguntar a laura)

            data_training = TensorDataset(distances_array_sub)
            loader = DataLoader(data_training, batch_size=500, shuffle=True)

            #dist, class_, jk, w
            for x in loader:  
                x = x[0]

                d, dclass, jk, w = x[:,0], x[:,1], x[:,2], x[:,3]
                optimizer.zero_grad()
                c = clustnet(d.unsqueeze(1).cuda(), jk.type(torch.LongTensor).cuda())#

                loss = CELoss(c.squeeze(1),dclass.type(torch.LongTensor).cuda())
                wloss = (w.cuda()*loss).mean()

                wloss.backward()
                optimizer.step()
            scheduler.step()
            print(wloss.item())
            
    def pred_clustering(self, min_sep= 0.03, max_sep = 26, nedges=8):
        clustnet=self.net_2pcf
        th_test = np.linspace(np.log10(min_sep), np.log10(max_sep), nedges)
        thetac_test = 10**np.array([(th_test[i]+th_test[i+1])/2 for i in range(len(th_test)-1)])
        thetac_test = thetac_test/60
        inp_test = torch.Tensor(thetac_test)
        
        preds=np.empty(shape=(400,7,2))
        for jk in range(400):
            jkf = torch.LongTensor(jk*np.ones(shape=inp_test.shape))
            c = clustnet(inp_test.unsqueeze(1).cuda(),jkf.cuda())
            s = nn.Softmax(1)
            p = s(c).detach().cpu().numpy()
            preds[jk]=p
            
        pred_ratio = preds[:,:,0]/(1-preds[:,:,0])-1
        return pred_ratio
    
