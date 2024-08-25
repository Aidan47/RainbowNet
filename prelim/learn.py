from model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
    
if __name__ == "__main__":
    data = np.load('gameData.npz', allow_pickle=True)
    X_train, Y_train, X_test, Y_test = data['array1'], data['array2'], data['array3'], data['array4']
    
    model = CNN(64).float()             # initialize model
    criterion = nn.CrossEntropyLoss()   # choose the loss optimizer and stuff
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    epochs = 5
    for i in range(epochs):
        for j in range(X_train.size):
            x, y = torch.tensor(X_train[j]), torch.tensor(Y_train[j])   # single training example
            
            # forward
            output = model(x.float())
            loss = criterion(output, np.argmax(y))
            
            # backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if (i+1) % 1 == 0:
            print(f"Epoch: {i+1}    Loss: {loss}")
    
    torch.save(model, "model")