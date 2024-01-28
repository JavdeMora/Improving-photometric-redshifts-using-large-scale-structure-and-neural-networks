#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import torch
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler
import random

##########################################################################################################

#Network definition

##########################################################################################################
class network_dists(torch.nn.Module):
    def __init__(self, nhidden):
        super().__init__()
        self.inputlay = torch.nn.Sequential(nn.Linear(1 ,10),nn.LeakyReLU(0.5))
        self.emblayer = torch.nn.Embedding(400,5)

        params = np.linspace(15,100,nhidden)
        modules_up,modules_down = [],[]
        for k in range(nhidden-1):
            modules_up.append(nn.Linear(int(params[k]) ,int(params[k+1])))
            modules_up.append(nn.Dropout(0.1))
            modules_up.append(nn.LeakyReLU(0.1))  
       
            modules_down.append(nn.Linear(int(params[-(k+1)]) ,int(params[-(k+2)])))
            modules_down.append(nn.Dropout(0.1))
            modules_down.append(nn.LeakyReLU(0.1))  
            
        self.hiddenUP= nn.Sequential(*modules_up)
        self.hiddenDOWN= nn.Sequential(*modules_down)
        
        
        self.outlay = torch.nn.Sequential(nn.Linear(15 ,5),nn.LeakyReLU(0.1),nn.Linear(5, 2))       
        #self.LogSoftmax = nn.LogSoftmax(dim = 1)
       
        
    def forward(self, inp, jk):
        
        embjk = self.emblayer(jk)
        x = self.inputlay(inp)
        x = torch.concat((x,embjk),1)
        
        
        x = self.hiddenUP(x)
        
        x = self.hiddenDOWN(x)
        #print(x.shape)

        c = self.outlay(x)
        
        return c


# In[31]:


def _network_training(net,epochs, distances_array):
    optimizer = optim.Adam(net.parameters(), lr=1e-6)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1200, gamma=0.01)
    CELoss = nn.CrossEntropyLoss(reduction='none')
    Nobj = 100_000 #CAMBIAR A distances_array.shape[0]
    for epoch in range(epochs):
        print('starting epoch', epoch)

        distances_array_sub = distances_array[np.random.randint(0, distances_array.shape[0], N_obj)]

        data_training = TensorDataset(distances_array_sub)
        loader = DataLoader(data_training, batch_size=500, shuffle=True)

        #dist, class_, jk, w
        for x in loader:  
            x = x[0]

            d, dclass, jk, w = x[:,0], x[:,1], x[:,2], x[:,3]
            optimizer.zero_grad()
            c = net(d.unsqueeze(1).cuda(), jk.type(torch.LongTensor).cuda())#

            loss = CELoss(c.squeeze(1),dclass.type(torch.LongTensor).cuda())
            wloss = (w.cuda()*loss).mean()

            wloss.backward()
            optimizer.step()
        scheduler.step()
        print(wloss.item())
    

