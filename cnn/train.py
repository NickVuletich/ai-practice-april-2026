# Programmer: Nicholas Vuletich
# Date: 4-4-2026
# File: train.py

import torch
import torch.nn as nn
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # model
    model = CNN().to(device)

    # training data
    X = torch.randn(100, 1, 8, 8).to(device)
    Y = (X.sum(dim=(2,3), keepdim=False) > 0).float().to(device)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    for epoch in range(10):
        output = model(X)
        loss = loss_fn(output, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item():.2f}")





if __name__ == "__main__":
    train_model()