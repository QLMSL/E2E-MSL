import torch
import torch.nn as nn

class SigmoidThreshold(nn.Module):
    def __init__(self, tau=0.01, sharpness=1000.0):
        super().__init__()
        self.tau = tau
        self.sharpness = sharpness

    def forward(self, x):
        return torch.sigmoid((x - self.tau) * self.sharpness)