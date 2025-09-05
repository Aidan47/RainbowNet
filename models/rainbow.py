from math import log2
from pathlib import Path
import torch
import torch.nn as nn
from .sae import SAE
from .translator import Translator
from .mlp import MLP


Root = Path().resolve().parent


class Rainbow(nn.Module):
    def __init__(self, input: int, latent_factor: int, output: int, layer_norm: bool, topk:bool):
        super(Rainbow, self).__init__()
        
        # get layer dimensions
        _, latent, decoder_H = self.get_dimensions(input, latent_factor)
        
        # make models
        self.encoder = self.load_encoder(input, latent_factor, layer_norm)
        self.mlp = MLP(latent, 3)
        self.translator = Translator(latent, decoder_H, output)
        self.topk = topk
        
        
    def get_dimensions(self, input, latent_factor):
        next_pow = pow(2, int(log2(input)) + 1)
        latent = latent_factor * next_pow
        decoder_H = latent // 2
        return next_pow, latent, decoder_H
    
    def load_encoder(self, input, latent_factor, layer_norm):
        sae = SAE(input, latent_factor, layer_norm)
        path = Root / f"checkpoints/sae"
        model = torch.load(path / "model.pth")
        sae.load_state_dict(model)
        sae.requires_grad_(False)
        return sae
        
    def type(self, x):
        if type(x) is not torch.Tensor:
            x = torch.from_numpy(x).float()
        return x
    
    def forward(self, x):
        x = type(x)
        x = self.encoder.encode(x, self.topk)
        x = self.mlp.forward(x)
        mean, log_std = self.translator.forward(x)
        return mean, log_std