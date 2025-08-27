import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, latent: int, layers: int):
        super(MLP, self).__init__()
        
        self.layers = []
        for _ in range(layers):
            self.layers.append(nn.Linear(latent, latent))
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x