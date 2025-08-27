import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Decoder(nn.Module):
    def __init__(self, latent: int, h: int, output: int, **kwargs):
        super(Decoder, self).__init__()
        
        self.l1 = nn.Linear(latent, h)
        self.l2 = nn.Linear(h, h)
        # pretraining head
        self.l3 = nn.Linear(h, output)
        # stochastic head for SAC
        self.mean = nn.Linear(h, output)
        self.std = nn.Linear(h, output)
        
    def forward(self, x, is_pretraining):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if is_pretraining:
            return self.l3(x)
        return self.mean(x), self.std(x)