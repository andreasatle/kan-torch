import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from Tutorial.kan_1xn import Kan1xN

# Generate some example data
def exact_y_solution(x):
    return 3.0 - torch.sqrt(x) + torch.sin(x)

def exact_z_solution(x):
    return 3.0 + torch.sqrt(x) - torch.cos(x)

def exact_zz_solution(x):
    return 2.0 + 0.5*torch.sqrt(x) + 0.7*torch.cos(x-0.2)

x_data = torch.linspace(3, 13, 100)
exact_solution = torch.stack([exact_y_solution(x_data), exact_z_solution(x_data), exact_zz_solution(x_data)], dim=0)
xx_data = torch.stack([x_data,x_data,x_data],dim=0)
y_data = exact_solution + torch.randn_like(exact_solution) * 0.1

# Instantiate the Kan1xN model
xa, xb = torch.min(x_data), torch.max(x_data)
dx = (xb - xa) * 0.01
kan1xN = Kan1xN(n_out=3, n_params=8, degree=3, lb=xa-dx, ub=xb+dx)

# Define the objective function
def objective_function(y_pred, y_true):
    return torch.sum((y_pred - y_true) ** 2)

# Define the optimizer
optimizer = optim.Adam(kan1xN.parameters(), lr=0.01)

# Training loop
num_epochs = 500
num_batches = 5
for epoch in range(num_epochs):
    for batch in range(num_batches):
        x_data_batch = x_data[batch::num_batches]
        y_data_batch = y_data[:,batch::num_batches]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass, always use the same `x_data`
        y_pred_batch = kan1xN(x_data_batch)

        # Compute loss
        loss = objective_function(y_pred_batch, y_data_batch)

        # Backward pass
        loss.backward()

        # Update spline coeffs
        optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the data and the fitted spline
knot_data = torch.stack([kan1xN.knots[2:-2], kan1xN.knots[2:-2], kan1xN.knots[2:-2]],dim=1)
plt.scatter(xx_data, y_data, label='Data')
plt.scatter(knot_data, kan1xN(kan1xN.knots[2:-2]).detach().numpy(), color='red', label='Fitted knots')
plt.plot(xx_data.T, kan1xN(x_data).detach().numpy().T, color='red', label='Fitted model')
plt.plot(xx_data.T, exact_solution.detach().numpy().T, color='green', label='Exact solution')
plt.legend()
plt.show()