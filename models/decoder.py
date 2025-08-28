import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Decoder(nn.Module):
    def __init__(self, latent: int, output: int, **kwargs):
        super(Decoder, self).__init__()
        
        self.l1 = nn.Linear(latent, output)
        
    def forward(self, x):
        return self.l1(x)