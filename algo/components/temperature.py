import torch

class Temperature:
    def __init__(self, lr, state_dim):
        self.entropy_target = -state_dim
        self.log_temp = torch.zeros(1, requires_grad=True)
        self.optimizer = torch.optim.Adam(self.log_temp, lr)
        
    def get_temp(self):
        return torch.exp(self.log_temp)
        
    def update(self, log_probability):
        temp_loss = -(self.log_temp * (log_probability + self.entropy_target)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        temp_loss.backward()
        self.optimizer.step()