import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

#define class 
class FC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #Define layers. 3 linear FC layers, ReLU to not limit gradient descent step sizes, dropout to promote generalization.
        #Don't know what width to make the network, or how many neurons to give as input/output
        self.layers = nn.Sequential(
            nn.Linear(1,6),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(6,6),
            nn.ReLU(),
            nn.Linear(6,1)
        )

    def forward(self, x):
        #Forward pass
        x = torch.sigmoid(self.layers(x))
        return x