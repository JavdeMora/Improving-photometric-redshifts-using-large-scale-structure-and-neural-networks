import numpy as np 
import torch
from torch import nn

class MTL_network(torch.nn.Module):
    def __init__(self, nhidden, num_gauss):
        super().__init__()
        self.inputlay = torch.nn.Sequential(nn.Linear(7, 20),nn.LeakyReLU(0.1))
        
        params = np.linspace(20,200,nhidden)
        modules = []
        for k in range(nhidden-1):
            modules.append(nn.Linear(int(params[k]) ,int(params[k+1])))
            #Do I do the up and down?
            #Do I add a Dropout layer? solo en clustering
            #Do I add weights?
            #Or do I leave it as the photoz network is configurated for proper comparison?
            modules.append(nn.LeakyReLU(0.1))  
        self.hiddenlay = nn.Sequential(*modules)
        
        self.dpred = torch.nn.Sequential(nn.Linear(200 ,100), nn.LeakyReLU(0.1),nn.Dropout(0.01), nn.Linear(100 ,50), nn.LeakyReLU(0.1), nn.Linear(50 ,15), nn.LeakyReLU(0.1),nn.Linear(15, 2)) 
        self.logalphas = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, num_gauss))                
        self.means = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100, num_gauss))
        self.logstds = torch.nn.Sequential(nn.Linear(200 ,100),nn.LeakyReLU(0.1),nn.Linear(100 ,num_gauss))
        
    def forward(self, inp):
        
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        mu = self.means(x)
        logsig = self.logstds(x)
        logalpha=self.logalphas(x)
        dpred = self.dpred(x)
        
        logsig = torch.clamp(logsig,-5,5)
        logalpha = logalpha - torch.logsumexp(logalpha,1)[:,None] 


        
        return logalpha, mu, logsig, dpred
