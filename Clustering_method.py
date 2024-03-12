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
    def __init__(self, 
                 min_sep,= 0.03
                 max_sep= 26,
                 nedges= 8,
                 cluster_hlayers, 
                 epochs, 
                 lr=1e-5 ,
                 batch_size = 500, 
                 pathfile_distances='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/d_100deg2_z0506_v2.npy', 
                 pathfile_drand='/data/astro/scratch2/lcabayol/EUCLID/MTL_clustering/dr_100deg2_v2.npy'
                ):
                    
        self.net_2pcf= network_dists(cluster_hlayers).cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_sep= min_sep
        self.max_sep= max_sep
        self.nedges= nedges
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.d, self.drand = self._load_distances_array()
        
        
    def _load_distances_array(self,  pathfile_distances, pathfile_drand):
       """
        Loads real and random distances from files.

        Args:
            pathfile_distances (str): Path to the distances file.
            pathfile_drand (str): Path to the random distances file.
            
        Returns:
            numpy array: real distances for a sky area divided in 400 jacknifes
            numpy array: random distances for a sky area divided in 400 jacknifes
            
        """ 
        #Load distances arrays
        d = np.load(pathfile_distances)#[:,:100] <-- if you want to test #(400, 100000)
        drand = np.load(pathfile_drand)#[:,:100] <-- if you want to test
        
        #Define angular separation limits and number of edges
        return d, drand
        
    def _get_distances_array(self):
        """
        Loads distances and random distances arrays from files.

        Args:
            pathfile_distances (str): Path to the distances file.
            pathfile_drand (str): Path to the random distances file.
            
        Returns:
            numpy array: of randomly ordered buckets with [distance value, true/random distance classification, jacknife, weight]
        """
        #Call distances arrays
        d = self.d
        drand = self.drand
        #Flatten array 
        distA = d.flatten()
        distB = drand.flatten()
        
        #Define angular separation limits and number of edges
        th = np.linspace(np.log10(self.min_sep), np.log10(self.max_sep), self.nedges)
        theta = 10**th * (1./60.)
        
        #Compute weights
        Ndd = np.array([len(distA[(distA>theta[k])&(distA<theta[k+1])]) for k in range(len(theta)-1)])
        Ndd = np.append(Ndd,len(distA[(distA>theta[-1])]))
        Pdd = Ndd / Ndd.sum()
        wdd = Pdd.max() / Pdd 
        #Compute ranges
        ranges =  [(theta[k],theta[k+1]) for k in range(len(theta)-1)]
        ranges.append((theta[-1],1.1))
        ranges = np.array(ranges)
        weights = wdd
        
        # Labeling arrays
        arr1 = d.copy().T
        arr2 = drand.copy().T
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
        
        #Adding weights to each record
        range_idx = np.searchsorted(ranges[:, 1], distances_array[:,0], side='right')
        w = weights[range_idx]
        distances_array = np.c_[distances_array, w.reshape(len(w),1)]
        distances_array = torch.Tensor(distances_array)
        
        return distances_array
    
    def train_clustering(self, Nobj='all', *args):
        """
        Train the clustering prediction model.
         Args:
            Nobj (float): Preguntar a Laura.
            *args: Additional arguments.

        Returns:
            None
        """
        #Call distances array
        distances_array=self._get_distances_array()
        # Transfer model to GPU
        self.clustnet= self.net_2pcf.cuda()
        
        # Define optimizer, learning rate scheduler and loss function
        optimizer = optim.Adam(clustnet.parameters(), lr=self.lr)# deberia separar entre lr photoz y lr clustering
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1200, gamma=0.01)
        CELoss = nn.CrossEntropyLoss(reduction='none')
        if Nobj=='all':
            Nobj = distances_array.shape[0]
        else:
            Nobj = float(Nobj)
        # Training loop
        for epoch in range(self.epochs):#deberia separar entre epochs photoz y epochs clustering
            print('starting epoch', epoch)
            #Creating loader
            distances_array_sub = distances_array[np.random.randint(0, distances_array.shape[0], distances_array.shape[0])]#revisar la size (yo he usado todas las distancias para entrenar, preguntar a laura)
            data_training = TensorDataset(distances_array_sub)
            loader = DataLoader(data_training, batch_size=self.batch_size, shuffle=True)

            # iterating for each element on the distances array: dist, class, jk, w
            for x in loader:  
                x = x[0]
                d, dclass, jk, w = x[:,0], x[:,1], x[:,2], x[:,3]
                optimizer.zero_grad()
                c = self.clustnet(d.unsqueeze(1).cuda(), jk.type(torch.LongTensor).cuda())#
                #Computing loss
                loss = CELoss(c.squeeze(1),dclass.type(torch.LongTensor).cuda())
                wloss = (w.cuda()*loss).mean()
                # Backpropagation and optimization
                wloss.backward()
                optimizer.step()
            # Update learning rate
            scheduler.step()
            #Print training loss
            print(wloss.item())
            
    def pred_clustering(self, theta_test):
        """
        Predict redshift using minimum and maximum separation inputs and the number of edges.

        Args:
            plot (bool): Whether to plot the predicted redshift distribution. Default is True.

        Returns:
            None
        """
        clustnet=self.net_2pcf.eval()
        inp_test = torch.Tensor(theta_test)

        #Create empty array for predictions
        preds=np.empty(shape=(400,7,2))
        #Make prediction with the inputs for each jacknife
        for jk in range(400):
            jkf = torch.LongTensor(jk*np.ones(shape=inp_test.shape))
            c = clustnet(inp_test.unsqueeze(1).cuda(),jkf.cuda())
            s = nn.Softmax(1)
            p = s(c).detach().cpu().numpy()
            preds[jk]=p
        #Computing 2PCF    
        pred_ratio = preds[:,:,0]/(1-preds[:,:,0])-1

        return pred_ratio
    
