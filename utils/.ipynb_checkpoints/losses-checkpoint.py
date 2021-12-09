import torch.nn as nn
import torch
class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        yhat = torch.tensor(yhat)
        y = torch.tensor(y)
        return torch.sqrt(torch.mean((yhat - y)**2))