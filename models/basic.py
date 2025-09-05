import numpy as np
import torch
import torch.nn as nn


def ortho_(l: torch.nn.Linear, gain: float):
    torch.nn.init.orthogonal_(l.weight, gain)
    if l.bias is not None:
        torch.nn.init.constant_(l.bias, 0.0)


class actor(nn.Module):
    def __init__(self, input: int, hidden, output: int, **kwargs):
        super(actor, self).__init__()
        ortho = kwargs.get("ortho", True)

        self.l1 = nn.Linear(input, hidden, dtype=torch.float32)
        self.l2 = nn.Linear(hidden, hidden, dtype=torch.float32)
        self.mean = nn.Linear(hidden, output, dtype=torch.float32)
        self.log_std = nn.Linear(hidden, output, dtype=torch.float32)

        # orthogonalize layers
        if ortho:
            ortho_(self.l1, np.sqrt(2))
            ortho_(self.l2, np.sqrt(2))
            ortho_(self.mean, 0.01)
            nn.init.constant_(self.log_std.weight, 0.0)
            nn.init.constant_(self.log_std.weight, -0.5)
        # layer normalization
        self.layer_norm = kwargs.get("layer_norm", True)


    def forward(self, x):
        x = nn.functional.silu(self.l1(x))
        x = nn.functional.silu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mean, log_std
    
    
    
class Basic(nn.Module):
    def __init__(self, input: int, **kwargs):
        super(Basic, self).__init__()
        hidden = kwargs.get("hidden", 256)

        self.l1 = nn.Linear(input, hidden, dtype=torch.float32)
        self.l2 = nn.Linear(hidden, hidden, dtype=torch.float32)
        self.l3 = nn.Linear(hidden, 1, dtype=torch.float32)
        self.norm = nn.LayerNorm(hidden)
        

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)

        # maybe seperate norm to make it optional
        x = nn.functional.silu(self.norm(self.l1(x)))
        x = nn.functional.silu(self.norm(self.l2(x)))
        return self.l3(x)