'''

actor class
initialize different models
based on constructor input


'''

import numpy as np
import torch
import torch.nn as nn
from models.rainbow import Rainbow
from models.basic import actor


class Actor(nn.Module):
    def __init__(self, model, cfg):
        self.model = self.make(model, cfg)
        self.scale = cfg["action_scale"]
        self.optimizer = torch.optim.Adam(model, lr=cfg["lr"])
        
        self.epsilon = torch.zeros(1)
        self.tanh_u = torch.zeros(1)
        
        
    def make(self, model, cfg):
        if model == "basic":
            return actor(**cfg["model"])
        else:
            return Rainbow(**cfg["model"])
        
    
    def sample(self, mean, log_std):
        std = log_std.exp()
        epsilon = torch.randn_like(mean)
        u = mean + std * epsilon
        self.tanh_u = torch.tanh(u)
        a = self.scale * self.tanh_u
        return a
    
    
    def log_probability(self, mean, log_std):
        log_prob = torch.zeros(1)
        # compute log-probablity
        logP_u = -0.5 * ((self.epsilon**2) + 2 * log_std + np.log(2 * np.pi))
        logP_u = logP_u.sum(dim=-1, keepdim=True)
        correction = torch.log(1 - self.tanh_u.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        act_dim = mean.shape[-1]
        log_prob = logP_u - correction - act_dim * np.log(self.scale)
        return log_prob
    
        
    def forward(self, state, deterministic, with_prob):
        mean, log_std = self.model.forward(state)
        if deterministic:
            return mean
        
        a = self.sample(mean, log_std)
        
        if with_prob:
            log_prob = self.log_probability(mean, log_std)
            return a, log_prob
        
        return a
        
        
    def update(self, states, critic1, critic2, log_temp):
        actions, log_prob = self.forward(states, False, True)
        qMin = torch.min(critic1.forward(states, actions), critic2.forward(states, actions))
        loss = torch.mean(log_temp.exp() * log_prob - qMin)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return log_prob