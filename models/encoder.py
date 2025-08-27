import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input: int, h: int, latent: int):
        super(Encoder, self).__init__()
        
        self.l1 = nn.Linear(input, h)
        self.l1 = nn.Linear(h, h)
        self.l2 = nn.Linear(h, latent)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        return F.relu(self.l2(x))