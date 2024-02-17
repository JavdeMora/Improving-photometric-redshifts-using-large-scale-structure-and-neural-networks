import numpy as np 
import torch
from torch import nn

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
