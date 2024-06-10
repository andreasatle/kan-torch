import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from Tutorial.kan_cascade import KanCascade

def param_dump(model):
    for name, par in model.named_parameters():
        print(name, par.data)

housing = fetch_california_housing()

X_train, X_test, Y_train, Y_test = train_test_split(
    housing.data, housing.target, test_size=0.25, random_state=42)

x_train = torch.tensor(X_train.T, dtype=torch.float32)
y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(1,-1)
x_test = torch.tensor(X_test.T, dtype=torch.float32)
y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(1,-1)

lb = torch.min(x_train,dim=1).values
ub = torch.max(x_train,dim=1).values
lb, ub = lb - 0.1 * (ub - lb), ub + 0.1 * (ub - lb)
kanCascade = KanCascade(ns=[8,4,2,1], n_params=24, degree=3, lb=lb, ub=ub)


# Define the optimizer
optimizer = optim.Adam(kanCascade.parameters(), lr=0.01)
criterion = nn.MSELoss()

def train_closure(x,y):
    def inner_closure():
        optimizer.zero_grad()  # Clear the gradients
        y_pred = kanCascade(x)  # Forward pass
        loss = criterion(y_pred, y)  # Compute the loss
        loss.backward()  # Backward pass
        return loss
    return inner_closure

def loss_closure(x, y):
    y_pred = kanCascade(x)
    return criterion(y_pred, y)

# Training loop
num_epochs = 300
num_batch = 25
for epoch in range(num_epochs):
    for batch in range(num_batch):
        x_train_batch = x_train[:,batch::num_batch]
        y_train_batch = y_train[:,batch::num_batch]
        # Zero the gradients
        optimizer.step(train_closure(x_train_batch, y_train_batch))

    # Print progress
    if (epoch + 1) % 1 == 0:
        train_loss = loss_closure(x_train, y_train)
        test_loss = loss_closure(x_test, y_test)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

y_pred = kanCascade(x_test)
loss = loss_closure(x_test, y_test)
print(f"Test Loss: {loss.item():.4f}")
