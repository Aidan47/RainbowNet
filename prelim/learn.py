import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

def process(data): # Process the data
    X = data[:, 0].reshape(-1, 1)
    
    # Convert Y values to one-hot encoding
    Y = np.zeros((len(data), 3))
    pos = [games.index(i) for i in data[:, 1]]
    Y[np.arange(len(data)), pos] = 1
    
    # Split the data into training and testing sets
    X_train, Y_train = X[:int(len(data)*0.8)], Y[:int(len(data)*0.8)]
    X_test, Y_test = X[int(len(Y)*0.8):], Y[int(len(data)*.8):]
    
    return X_train, Y_train, X_test, Y_test
    
    
if __name__ == "__main__":
    games = ["Humanoid-v4", "HumanoidStandup-v4", "Hopper-v4"]
    data = np.load("gameData.npy", allow_pickle=True)
    
    X_train, Y_train, X_test, Y_test = process(data)
    
    dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # initialize model
    '''
    model = ...
    '''
    
    # choose the loss optimizer and stuff
    '''
    criterion = nn.CrossEntropyLoss()
    '''
    
    epochs = 100
    for i in range(epochs):
        for x, y in dataloader:
            # forward
            # output = model(x)
            # loss = criterion(output, y)
            
            # backward
            '''
                    review pytorch syntax & its motivation (YT Videos)
            '''
            pass
        pass
    