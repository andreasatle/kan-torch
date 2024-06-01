import torch
import torch.nn as nn
from Tutorial.kan_1xn import Kan1xN
import matplotlib.pyplot as plt
import numpy as np

class KanMxN(nn.Module):
    def __init__(self, n_in, n_out, n_params, degree, lb=None, ub=None):
        super(KanMxN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_params = n_params
        self.degree = degree
        if lb is not None:
            self.lb = lb
        else: 
            self.lb = torch.zeros(n_in)
        if ub is not None:
            self.ub = ub
        else:
            self.ub = torch.ones(n_in)
        self.kan1xNs = nn.ModuleList([Kan1xN(n_out=n_out, n_params=n_params, degree=degree, lb=self.lb[i], ub=self.ub[i]) for i in range(n_in)])

    def forward(self, x):
        out_list = torch.stack([self.kan1xNs[i](x[i]) for i in range(self.n_in)])
        return torch.sum(out_list, dim=0)

    #def plot(self):
    #    idx = 0
    #    for i in range(self.n_in):
    #        lb, ub = self.lb[i], self.ub[i]
    #        x = torch.linspace(lb, ub, 100)
    #        y = self.kan1xNs[i](x).detach().numpy()
    #        for j in range(self.n_out):
    #            idx += 1
    #            plt.subplot(self.n_in, self.n_out, idx)
    #            plt.plot(x, y[j], label=f"y_{j+1}") 
    #    plt.show()
