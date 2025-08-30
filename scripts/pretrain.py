import sys
import torch
import torch.nn.functional as F
import numpy as np
from models.sae import SAE
from scripts.data import DataLoader



def eval(sae, dataset, topk, batch_size=2048):
    batch = dataset.sample(batch_size, dataset='Val')
    predicted = sae.forward(batch, topk=topk)
    return F.mse_loss(predicted, batch)


def train(dataset, sae, topk, lr=1e-3, weight_decay=1e-4, beta=5e-6, batch_size=2048, trained=False):
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = last_avg_val = float('inf')
    losses = []
    val_losses = []
    timer = 5
    step = 1

    while not trained:
        batch = dataset.sample(batch_size)
        y = sae.forward(batch, topk=topk)
        
        # compute loss
        recon_loss = F.mse_loss(y, batch)
        l1 = sae.l1_reg()
        with torch.no_grad():
            l0_norm = (sae.z != 0).float().sum(dim=1).mean().item()
        loss = recon_loss
        if not topk:
            loss += beta * l1
        
        # update sae
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            losses.append(loss.detach().numpy())
            val_losses.append(eval(sae, dataset, topk))
        if val_losses[-1] <= best_loss:
            best_loss = val_losses[-1]
        
                
        if step % 50 == 0:
            # early stopping on average val loss
            avg = np.mean(losses[-50:])
            avg_val= np.mean(val_losses[-50:])
            if avg_val <= last_avg_val:
                last_avg_val = avg_val
                timer = 5
                sae.save()
            else:
                timer -= 1
                if timer == 0:
                    trained = True
                    
            # print info
            print(f"step: {step}, loss: {avg:.2f}, val loss: {avg_val:.2f}, l0 norm: {l0_norm:.1f}")
        step += 1
        
    sae.save()
    
    
if __name__ == "__main__":
    # if using topk
    topk = True if sys.argv[1].lower() == "true" else False
    # if training on pre/post norm states
    norm = True if sys.argv[2].lower() == "true" else False
    
    # initialize models & dataset
    sae = SAE(348, 4, 348, layer_norm=norm)
    dataset = DataLoader(layer_norm=norm)
    
    train(dataset, sae, topk)