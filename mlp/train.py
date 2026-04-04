# Programmer: Nicholas Vuletich
# Date: 4-4-2026
# File: train.py

import torch
import torch.nn as nn
from model import MLP

# Sets device to cuda (aka. GPU) or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    model = MLP().to(device)

    # training data
    X = torch.randn(100, 8).to(device)
    Y = (X.sum(dim=1, keepdim=True) > 0).float().to(device)

    batch_size = 32

    # Test data
    test_X = torch.randn(32, 8).to(device)
    test_Y = (test_X.sum(dim=1, keepdim=True) > 0).float().to(device)

    # Backprop
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_num = 0

    for epoch in range(10):

        total_batches = 0
        total_loss = 0

        for i in range(0, len(X), batch_size):
            total_batches += 1
            
            # sets batch from i to batchsize
            batch_x = X[i: i + batch_size]
            batch_y = Y[i: i + batch_size]

            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # testing the model with test data
        with torch.no_grad():
            print(f"Test Number: {test_num}.")
            test_num += 1

            # for accuracy
            total_correct = 0
            total_iterations = 0

            # gets prediction from model
            preds = model(test_X)

            # turns tensor into 0's and 1's based on float number
            preds = (preds > 0.5).float()

            # gets predictions and labels and computes accuracy
            for p, l in zip(preds, test_Y):
                p = p.item()
                l = l.item()

                if p == l:
                    total_correct += 1

                total_iterations += 1
            
            accuracy = (total_correct/total_iterations) * 100

            print(f"Accuracy: {accuracy:.2f}%.")

        epoch_loss = total_loss/total_batches

        print(f"Loss: {epoch_loss:.4f}")
    



if __name__ == "__main__":
    train_model()
