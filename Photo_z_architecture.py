
import torch
import torch.nn as nn
import numpy as np

class photoz_network(torch.nn.Module):
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
