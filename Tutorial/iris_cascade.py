import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from Tutorial.kan_cascade import KanCascade

# Set this for debugging!
torch.autograd.set_detect_anomaly(True)

iris = load_iris()
X, y = iris.data.T, iris.target
x_data = torch.tensor(X, dtype=torch.float32)
y_data = torch.tensor(y, dtype=torch.long)
lb = torch.min(x_data,dim=1).values-0.1
ub = torch.max(x_data,dim=1).values+0.1


# Instantiate the KanCascade model
kanCascade = KanCascade(ns=[4,4,3,3], n_params=8, degree=3, lb=lb, ub=ub)
# Define the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(kanCascade.parameters(), lr=0.01)
def closure(x,y):
    def inner_closure():
        optimizer.zero_grad()  # Clear the gradients
        y_pred = kanCascade(x)  # Forward pass
        loss = criterion(y_pred.T, y)  # Compute the loss
        loss.backward()  # Backward pass
        return loss
    return inner_closure

def param_dump(model):
    for name, par in model.named_parameters():
        print(name, par.data)

#param_dump(kanCascade)

# Training loop
num_epochs = 200
num_batch = 3
for epoch in range(num_epochs):
    for batch in range(num_batch):
        x_data_batch = x_data[:,batch::num_batch]
        y_data_batch = y_data[batch::num_batch]
        # Zero the gradients
        optimizer.step(closure(x_data_batch, y_data_batch))
        kanCascade.plot()
    # Print progress
    if (epoch + 1) % 10 == 0:
        loss = closure(x_data, y_data)()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        

#kanCascade.plot()
y_pred = kanCascade(x_data)

print(f"Misclassified: {torch.sum(torch.argmax(y_pred, dim=0)!=y_data)} / {y_data.size(0)}")
param_dump(kanCascade)
