import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Translator(nn.Module):
    def __init__(self, latent: int, h: int, output: int):
        super(Translator, self).__init__()
        
        self.l1 = nn.Linear(latent, h)
        self.l2 = nn.Linear(h, h)
        self.mean = nn.Linear(h, output)
        self.std = nn.Linear(h, output)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.mean(x), self.std(x)