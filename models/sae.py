from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .encoder import Encoder
from .decoder import Decoder


class SAE(nn.Module):
    def __init__(self, input: int, latent_factor: int, output: int):
        super(SAE, self).__init__()
        
        # get layer dimensions
        h1 = pow(2, int(log2(input)) + 1)
        h2 = 2 * h1 if latent_factor > 2 else 0
        latent = latent_factor * h1
        
        # make models
        self.encoder = Encoder(input, h1, h2, latent, norm=True)
        self.decoder = Decoder(latent, input)

        self.z = torch.Tensor([])
        
    def type(self, x):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        return x
    
    def topk(self, x, k=64):
        top_k_values, _ = torch.topk(x, k)
        kth_value = top_k_values[:, -1].unsqueeze(-1)
        # a mask to zero out non-top-k values
        mask = (x >= kth_value).float()
        return x * mask
        
    
    def forward(self, x, **kwargs):
        topk = kwargs.get("topk", True)
        x = self.type(x)
        x = self.z = self.encoder.forward(x)
        if topk:
            x = self.z = self.topk(x)
        x = self.decoder.forward(x)
        return x
    
    def encode(self):
        # forward without decoder
        pass
    
    def l1_reg(self):
        return torch.sum(torch.abs(self.z))
    
    def l2_reg(self):
        W2 = torch.square(self.decoder.l1.weight)
        return W2.sum()
    
    def kl_penalty(self, target_sparsity):
        rho_hat = torch.mean(self.z)  # average activation per unit
        print(rho_hat.shape)
        rho = torch.full_like(rho_hat, target_sparsity)
        
        term1 = rho * torch.log(rho / rho_hat.clamp(min=1e-8))
        term2 = (1 - rho) * torch.log((1 - rho) / (1 - rho_hat).clamp(min=1e-8))
        kl_div = term1 + term2
        
        return torch.sum(kl_div)
    
    def save(self):
        torch.save(self.encoder.state_dict(), "checkpoints/sae/encoder.pth")
        torch.save(self.decoder.state_dict(), "checkpoints/sae/decoder.pth")        