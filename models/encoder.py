import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input: int, h1, h2, latent: int, layer_norm: bool):
        super(Encoder, self).__init__()
        
        self.l1 = nn.Linear(input, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, h2)
        self.l4 = nn.Linear(h2, latent)
        
        self.norm = layer_norm
        if self.norm:
            self.normalize = nn.LayerNorm(latent)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        if self.norm:
            x = self.normalize(x)
        return F.relu(x)