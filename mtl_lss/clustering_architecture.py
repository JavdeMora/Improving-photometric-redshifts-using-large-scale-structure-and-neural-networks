import numpy as np 
import torch
from torch import nn

class network_dists(torch.nn.Module):
    """
    A neural network designed for predicting the 2pcf of galaxies based on the distribution of data and random pairs.

    It consists of an input layer, an embedding layer, multiple hidden layers going up and down, and an output layer.
    
    Attributes:
        nhidden (int): Number of hidden layers in the network.
    """
    def __init__(self, nhidden):
        super().__init__()
        
        # Define the layers of the neural network
        self.inputlay = torch.nn.Sequential(nn.Linear(1 ,10), nn.LeakyReLU(0.5))
        self.emblayer = torch.nn.Embedding(400, 5)  # Embedding layer
        
        # Define the hidden layers
        params = np.linspace(15, 100, nhidden)
        modules_up, modules_down = [], []
        for k in range(nhidden - 1):
            # Layers for going up the network
            modules_up.append(nn.Linear(int(params[k]), int(params[k+1])))
            modules_up.append(nn.Dropout(0.1))  # Dropout layer to prevent overfitting
            modules_up.append(nn.LeakyReLU(0.1))  # LeakyReLU activation function
            
            # Layers for going down the network
            modules_down.append(nn.Linear(int(params[-(k+1)]), int(params[-(k+2)])))
            modules_down.append(nn.Dropout(0.1))  # Dropout layer
            modules_down.append(nn.LeakyReLU(0.1))  # LeakyReLU activation function
        
        # Define the sequential modules for the hidden layers going up and down
        self.hiddenUP = nn.Sequential(*modules_up)
        self.hiddenDOWN = nn.Sequential(*modules_down)
        
        # Output layer
        self.outlay = torch.nn.Sequential(nn.Linear(15 ,5), nn.LeakyReLU(0.1), nn.Linear(5, 2))       
        #self.LogSoftmax = nn.LogSoftmax(dim=1)  # LogSoftmax activation function for output layer
        
    def forward(self, inp, jk):
        # Forward pass of the neural network
        
        # Embedding for jk
        embjk = self.emblayer(jk)
        # Input layer
        x = self.inputlay(inp)
        # Concatenate input with jk embedding
        x = torch.concat((x, embjk), 1)
        
        # Pass through the hidden layers going up
        x = self.hiddenUP(x)
        # Pass through the hidden layers going down
        x = self.hiddenDOWN(x)

        # Output layer
        c = self.outlay(x)
        
        return c
