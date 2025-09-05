import numpy as np
import torch
import torch.nn as nn
from AI.Research.RainbowNet.models.basic import Basic 


class Critic(nn.Module):
    def __init__(self, model, cfg):
        super(Critic, self).__init__()

        self.model = self.make(model, cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])
    
    def make(self, model, cfg):
        # get proper class
        if model != "basic":
            # add other model initialization
            pass
        return Basic(**cfg)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.model.forward(state, action)
    
    def update_critics(self, y, states, actions):
        loss = nn.functional.mse_loss(self.model.forward(states, actions), y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()