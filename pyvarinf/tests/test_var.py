from unittest import TestCase

import pyvarinf
import torch
import torch.nn as nn

from torch.autograd import Variable

class TestVar(TestCase):
    def test_var_lin(self):
        x = Variable(torch.Tensor(1, 10).fill_(2))
        dx = torch.Tensor(1, 10).fill_(1)
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model)
        out = var_model(x)
        out.backward(dx)
        model = nn.Linear(10, 10, False)
        var_model = pyvarinf.Variationalize(model)
        out = var_model(x)
        out.backward(dx)

    def test_var_lay(self):
        x = Variable(torch.Tensor(1, 10).fill_(2))
        dx = torch.Tensor(1, 10).fill_(1)
        class SubModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.p = nn.Parameter(torch.Tensor(1, 10).fill_(1))

            def forward(self, input):
                return self.p * self.fc1(input)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.sub = SubModel()

            def forward(self, input):
                return self.fc1(self.sub(input))

        model = Model()
        var_model = pyvarinf.Variationalize(model)
        out = var_model(x)
        out.backward(dx)

    def test_diff(self):
        x = Variable(torch.Tensor(1, 10).fill_(2))
        dx = torch.Tensor(1, 10).fill_(1)
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model)
        out1 = var_model(x)
        out2 = var_model(x)
        self.assertTrue(out1.eq(out2).float().norm().data[0] == 0)
