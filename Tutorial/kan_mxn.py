import torch
import torch.nn as nn
from Tutorial.kan_1xn import Kan1xN
import matplotlib.pyplot as plt

class KanMxN(nn.Module):
    def __init__(self, n_in, n_out, n_params, degree, lb, ub):
        super(KanMxN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_params = n_params
        self.degree = degree
        self.lb = lb
        self.ub = ub
        self.kan1xNs = nn.ModuleList([Kan1xN(n_out=n_out, n_params=n_params, degree=degree, lb=lb[i], ub=ub[i]) for i in range(n_in)])

    def forward(self, x):
        out = torch.zeros((x.size(0), self.n_out), dtype=torch.float32)
        for i in range(self.n_in):
            out += self.kan1xNs[i](x[:,i])
        return out

    def plot(self):
        idx = 0
        for i in range(self.n_in):
            lb, ub = self.lb[i], self.ub[i]
            x = torch.linspace(lb, ub, 100)
            y = self.kan1xNs[i](x)
            for j in range(self.n_out):
                idx += 1
                plt.subplot(self.n_in, self.n_out, idx)
                plt.plot(x.detach().numpy(), y[:,j].detach().numpy(), label=f"y_{j+1}") 
        plt.show()
