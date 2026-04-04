# Programmer: Nicholas Vuletich
# Date: 4-4-2026
# File: model.py

import torch
import torch.nn as nn

# Example input for model (batch, 1, 8, 8)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 16,
            kernel_size = (3,3),
            stride = 1, 
            padding = 1
        )
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(
            in_features=1024,
            out_features=1
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.sig(x)
        return x