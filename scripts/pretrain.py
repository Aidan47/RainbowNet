import sys
import torch
import torch.nn.functional as F
from models.sae import SAE
from scripts.data import DataLoader



def eval(sae, dataset, topk, batch_size=512):
    batch = dataset.sample(batch_size, dataset='Val')
    predicted = sae.forward(batch, topk)
    return F.mse_loss(predicted, batch)


def train(dataset, sae, topk, lr=1e-3, weight_decay=1e-4, beta=5e-6, batch_size=1024, trained=False):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float('inf')
    timer = 100
    step = 0

    while not trained:
        batch = dataset.sample(batch_size)
        y = sae.forward(batch, topk=topk)
        
        # compute loss
        recon_loss = F.mse_loss(y, batch)
        l1 = sae.l1_reg()
        with torch.no_grad():
            l0_norm = (sae.z != 0).float().sum(dim=1).mean().item()
        loss = recon_loss + beta * l1
        
        # update sae
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # early stopping on validation loss
        val_loss = eval(sae, dataset, topk)
        if  val_loss <= best_loss:
            best_loss = val_loss
            timer = 20
            sae.save()
        else:
            timer -= 1
            if timer == 0:
                trained = True
                
        # print info
        if step % 20 == 0:
            print(f"epoch: {step}, loss: {recon_loss:.2f}, val loss: {val_loss:.2f}, l0 norm: {l0_norm:.1f}")
        step += 1
        
    sae.save()
    
    
if __name__ == "__main__":
    # if using topk
    topk = sys.argv[1]
    
    # initialize models & dataset
    sae = SAE(348, 2, 348)
    dataset = DataLoader()
    
    train(dataset, sae, topk)