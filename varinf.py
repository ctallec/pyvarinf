import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_w = Parameter(torch.zeros(10, 10))
        self.mean_b = Parameter(torch.zeros(10))
        self.rho_w = Parameter(torch.Tensor(10, 10).fill_(-1))
        self.rho_b = Parameter(torch.Tensor(10).fill_(-1))
        self.eps_w = Variable(torch.Tensor(10, 10))
        self.eps_b = Variable(torch.Tensor(10))

    def forward(self, input):
        self.eps_w.data.normal_()
        self.eps_b.data.normal_()
        weight = self.mean_w + (1+self.rho_w.exp()).log() * self.eps_w
        bias = self.mean_b + (1+self.rho_b.exp()).log() * self.eps_b
        return F.linear(input, weight, bias)

if __name__ == '__main__':
    input = Variable(torch.Tensor(10, 10).normal_())
    out = Variable(torch.Tensor(10, 10).normal_())

    model = Model()

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    for i in range(10000):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, out)

        print("{} {} -- approx log max std: {}".format(i, loss.data[0],
                                               max(model.rho_w.data.max(),model.rho_b.data.max())))
        loss.backward()
        optimizer.step()
