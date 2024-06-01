import torch
import torch.nn as nn
from Tutorial.kan_mxn import KanMxN

class KanCascade(nn.Module):
    def __init__(self, ns, n_params, degree, lb, ub):
        super(KanCascade, self).__init__()
        self.ns = ns
        self.n_params = n_params
        self.degree = degree
        self.lb = lb
        self.ub = ub

        bounds = lambda i, val: val if i == 0 else None
        self.sigmoid = nn.Sigmoid()
        self.kanMxNs = nn.ModuleList([KanMxN(n_in=ns[i], n_out=ns[i+1], n_params=n_params, degree=degree, lb=bounds(i,lb), ub=bounds(i,ub)) for i in range(len(ns)-1)])

    def forward(self, x):
        y = self.kanMxNs[0](x)
        for i in range(1, len(self.kanMxNs)):
            z = self.sigmoid(y)
            y = self.kanMxNs[i](z)
        return y