import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self, out_channels:int):
        super(CNN, self).__init__()
        
        self.convs = []
        self.convs.append(nn.Conv1d(1, out_channels, kernel_size=3, padding="same"))
        self.convs.append(nn.Conv1d(1, out_channels, kernel_size=5, padding="same"))
        
        self.fc = nn.Linear(out_channels * 2, 3)
        
    def forward(self, input):
        x = input.unsqueeze(1)
        
        outputs = []
        for conv in self.convs:
            x = F.relu(conv(x))
            outputs.append(F.adaptive_max_pool1d(x, 1).squeeze(-1))     # for takes max value from each filter/channel
        
        concatenated = torch.cat(outputs, 1)    # reducing all filters to one of the same dimension (matrix -> vector || 3d tensor -> matrix)
        
        logit = self.fc(concatenated)
        return F.softmax(logit, dim=1)