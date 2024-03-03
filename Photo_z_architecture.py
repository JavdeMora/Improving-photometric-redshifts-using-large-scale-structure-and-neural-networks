import torch
import torch.nn as nn
import numpy as np

class photoz_network(torch.nn.Module):
    """
    A neural network for predicting photo-z using a mixture density network.

    This network takes input features and predicts parameters for a Gaussian mixture model
    representing the photo-z distribution.

    Attributes:
        nhidden (int): Number of hidden layers in the network.
        num_gauss (int): Number of Gaussian components in the mixture model.
    """
    def __init__(self, nhidden, num_gauss):
        super().__init__()
        
        # Input layer
        self.inputlay = torch.nn.Sequential(nn.Linear(6, 20), nn.LeakyReLU(0.1))
        
        # Hidden layers
        params = np.linspace(20, 200, nhidden)
        modules = []
        for k in range(nhidden - 1):
            modules.append(nn.Linear(int(params[k]), int(params[k+1])))
            modules.append(nn.LeakyReLU(0.1))
        self.hiddenlay = nn.Sequential(*modules)
        
        # Output layers for mean, log standard deviation, and mixture weights
        self.means = torch.nn.Sequential(nn.Linear(200, 100), nn.LeakyReLU(0.1), nn.Linear(100, num_gauss))
        self.logstds = torch.nn.Sequential(nn.Linear(200, 100), nn.LeakyReLU(0.1), nn.Linear(100, num_gauss))
        self.logalphas = torch.nn.Sequential(nn.Linear(200, 100), nn.LeakyReLU(0.1), nn.Linear(100, num_gauss))
        
    def forward(self, inp):
        """
        Perform forward pass through the network.

        Args:
            inp (torch.Tensor): Input features.

        Returns:
            logalpha (torch.Tensor): Logarithm of the mixture weights.
            mu (torch.Tensor): Mean of the Gaussian components.
            logsig (torch.Tensor): Logarithm of the standard deviations of the Gaussian components.
        """
        # Forward pass through the layers
        x = self.inputlay(inp)
        x = self.hiddenlay(x)
        mu = self.means(x)
        logsig = self.logstds(x)
        logalpha = self.logalphas(x)
        
        # Clamp logsig to prevent extreme values
        logsig = torch.clamp(logsig, -5, 5)
        
        # Ensure the mixture weights sum up to 1 using softmax
        logalpha = logalpha - torch.logsumexp(logalpha, 1)[:, None]

        return logalpha, mu, logsig
