#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# In[16]:


##########################################################################################################

#Mixture Density Network

##########################################################################################################
class network(torch.nn.Module):
    def __init__(self, nhidden, num_gauss):
        super().__init__()
        self.inputlay = torch.nn.Sequential(nn.Linear(6, 20),nn.LeakyReLU(0.1))
        
        params = np.linspace(20,200,nhidden)
        modules = []
        for k in range(nhidden-1):
            modules.append(nn.Linear(int(params[k]) ,int(params[k+1])))
            modules.append(nn.LeakyReLU(0.1))  
        self.hiddenlay = nn.Sequential(*modules)
        
        
        self.logalphas = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, num_gauss))                
        self.means = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, num_gauss))
        self.logstds = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100 ,num_gauss))
        
    def forward(self, inp):
        
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        mu = self.means(x)
        logsig = self.logstds(x)
        logalpha=self.logalphas(x)
        
        logsig = torch.clamp(logsig,-5,5)
        
        
        logalpha = logalpha - torch.logsumexp(logalpha,1)[:,None] 


        
        return logalpha, mu, logsig  


# In[18]:


##########################################################################################################

# Network Training

##########################################################################################################
def _network_training(net,epochs):
    train_losses = [] 
    alpha_list = []
    mu_list = []
    ztrue_list = []
    optimizer = optim.Adam(net.parameters(), lr=2e-3) #, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) #para ir cambiando el lr a medida q se itera
    for epoch in range(epochs):
        for datain, xeval in loader_train:
            optimizer.zero_grad() 
            logalpha, mu, logsig= net(datain.to(device))
            sig = torch.exp(logsig)
            #loss function: 
            log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((xeval.to(device)[:,None] - mu[:,:,None])/sig[:,:,None])**2
            log_prob = torch.logsumexp(log_prob,1)
            loss = - log_prob.mean()
            loss.backward()
            optimizer.step()
       
            
            train_loss = loss.item()
            train_losses.append(train_loss)
        scheduler.step()
        
        net.eval()#desactivar algunas funciones de la red para poder evaluar resultados
        val_losses = []
        logalpha_list = []
        out_pred, out_true = [],[]
        with torch.no_grad():
            for xval, yval in loader_val:
                logalpha, mu, logsig= net(xval.to(device))
                sig = torch.exp(logsig)
                log_prob = logalpha[:,:,None] - logsig[:,:,None] - 0.5*((yval.to(device)[:,None] - mu[:,:,None])/sig[:,:,None])**2
                log_prob = torch.logsumexp(log_prob,1)
                loss = - log_prob.mean()

                val_loss = loss.item()
                val_losses.append(val_loss)
            
            if epoch % 1 == 0:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, epochs, train_loss, val_loss))

