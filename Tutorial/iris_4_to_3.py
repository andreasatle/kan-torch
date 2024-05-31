import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from Tutorial.kan_mxn import KanMxN

iris = load_iris()
X, y = iris.data, iris.target
x_data = torch.tensor(X, dtype=torch.float32)
y_data = torch.tensor(y, dtype=torch.long)
lb = torch.min(x_data,dim=0).values-0.1
ub = torch.max(x_data,dim=0).values+0.1

def param_dump(model):
    for name, par in model.named_parameters():
        print(name, par.data)

# Instantiate the KanMxN model
kanMxN = KanMxN(n_out=3, n_in=4, n_params=8, degree=3, lb=lb, ub=ub)
#param_dump(kanMxN)

# Define the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(kanMxN.parameters(), lr=0.01)

# Training loop
num_epochs = 500
num_batch = 5
for epoch in range(num_epochs):
    for batch in range(num_batch):
        x_data_batch = x_data[batch::num_batch]
        y_data_batch = y_data[batch::num_batch]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass, always use the same `x_data`
        y_pred = kanMxN(x_data_batch)

        # Compute loss
        loss = criterion(y_pred, y_data_batch)

        # Backward pass
        loss.backward()

        # Update spline coeffs
        optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

kanMxN.plot()
y_pred = kanMxN(x_data)

print(f"Misclassified: {torch.sum(torch.argmax(y_pred, dim=1)!=y_data)} / {y_data.size(0)}")
param_dump(kanMxN)
