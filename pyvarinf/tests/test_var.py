# pylint:disable=no-self-use
""" Test suite pyvarinf """
from unittest import TestCase

import pyvarinf
import torch
import torch.nn as nn

from torch.autograd import Variable


class TestVar(TestCase):
    """ Test suite for pyvarinf """

    def test_var_lin(self):
        """ Test linear model """
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
        """ Test imbrication """
        x = Variable(torch.Tensor(1, 10).fill_(2))
        dx = torch.Tensor(1, 10).fill_(1)

        class SubModel(nn.Module):
            """ Submodel """
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.p = nn.Parameter(torch.Tensor(1, 10).fill_(1))

            def forward(self, *inputs):
                return self.p * self.fc1(*inputs)

        class Model(nn.Module):
            """ Model """
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.sub = SubModel()

            def forward(self, *inputs):
                return self.fc1(self.sub(*inputs))

        model = Model()
        var_model = pyvarinf.Variationalize(model)
        out = var_model(x)
        out.backward(dx)

    def test_diff(self):
        """ Test that two consecutive evaluations don't
        yield the same result """
        x = Variable(torch.Tensor(1, 10).fill_(2))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model)
        out1 = var_model(x)
        out2 = var_model(x)
        self.assertTrue(out1.eq(out2).float().norm().data[0] == 0)
