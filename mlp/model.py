# Programmer: Nicholas Vuletich
# Date: 4-4-2026
# File: model.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=8,
            out_features=16
        )
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(
            in_features=16,
            out_features=1
        )
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.act(x)
        return x