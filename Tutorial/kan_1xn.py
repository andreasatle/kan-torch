import torch
import torch.nn as nn

class Kan1xN(nn.Module):
    def __init__(self, n_out, n_params, degree, lb, ub):
        super(Kan1xN, self).__init__()
        self.n_out = n_out
        self.n_params = n_params
        self.degree = degree
        self.lb = lb
        self.ub = ub
        self.knots = self.get_knots(lb, ub)

        self.params = nn.Parameter(torch.zeros(self.n_out, self.n_params))
        self.params.data = 0.1*torch.randn_like(self.params.data)
        
    def eval_basis(self, x):
        
        n_knots = len(self.knots)
        phis = torch.zeros((n_knots-1,len(x)), dtype=torch.float32)

        for i in range(n_knots-1):
            phis[i] = ((self.knots[i] <= x) & (x < self.knots[i+1])).float()

        for d in range(1, self.degree+1):
            phis2 = torch.zeros_like(phis)
            for i in range(n_knots-d-1):
                left_denom = self.knots[i+d] - self.knots[i]
                right_denom = self.knots[i+d+1] - self.knots[i+1]
                if left_denom != 0.0:
                    left_term = (x - self.knots[i]) / left_denom
                else:
                    left_term = torch.zeros_like(x)
                if right_denom != 0.0:
                    right_term = (self.knots[i+d+1] - x) / right_denom
                else:
                    right_term = torch.zeros_like(x)
                phis2[i] = left_term * phis[i] + right_term * phis[i+1]
            phis = phis2

        # Add the SiLU activation function as the last phi
        phis[n_knots-self.degree-1] = nn.SiLU()((x-self.lb)/(self.ub-self.lb))
        
        return phis[:n_knots-self.degree]

    def get_knots(self, a, b):
        return (b-a)*torch.cat((torch.zeros(self.degree),torch.linspace(0,1,self.n_params-self.degree),torch.ones(self.degree))) + a

    def forward(self, x):
        return self.params @ self.eval_basis(x)
