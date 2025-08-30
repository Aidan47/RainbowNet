import os
from pathlib import Path
import numpy as np
import torch



class DataLoader:
    def __init__(self, layer_norm, **kwargs):
        # dataset split
        file_size   = 1e5
        self.Train  = [0, int(kwargs.get("train", 0.9) * file_size)]
        self.Val    = [self.Train[1], self.Train[1]+int(kwargs.get("val", 0.05) * file_size)]
        self.Test   = [self.Val[1], self.Val[1]+int(kwargs.get("test", 0.05) * file_size)]
        
        HERE = Path(__file__).resolve().parent
        ROOT = HERE.parent
        norm_folder = "norm" if layer_norm else "pre_norm"
        self.path   = ROOT / f"datasets/{norm_folder}"
        self.idx    = 0    
        
    def get_files(self):
        F = os.listdir(self.path)
        F = [self.path / f for f in F]
        return F
    
    def load_data(self, f, mini_size, dataset):
        mini_batch = np.load(f, mmap_mode='r')
        mini_batch = mini_batch[dataset[0]:dataset[1]]
        return mini_batch[self.idx:self.idx+mini_size]
        
    def get_batch(self, mini_size, leftover, dataset):
        batch = np.array([[]]).reshape(0,348)
        F = self.get_files()
        for i, f in enumerate(F):
            mini_batch = np.array([[]]).reshape(0,348)
            if i < leftover:
                mini_batch = self.load_data(f, mini_size+1, dataset)
            else:
                mini_batch = self.load_data(f, mini_size, dataset)
            batch = np.vstack((batch, np.array(mini_batch)))
        
        return batch
    
    def sample(self, batch_size, **kwargs):
        # get dataset indices
        dataset =  self.__dict__[kwargs.get("dataset", 'Train')]
        
        # number of states per mini batch
        mini_size = batch_size // 20
        leftover = batch_size % 20
        
        batch = self.get_batch(mini_size, leftover, dataset)
        
        # compute next batch index
        if leftover:
            self.idx += (mini_size + 1) % len(dataset)
        else:
            self.idx += mini_size % len(dataset)
            
        # if another mini batch cant fit
        if len(dataset) - self.idx * mini_size < mini_size + 1:
            self.idx = 0
            
        return torch.from_numpy(batch).float()