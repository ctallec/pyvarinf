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

    def test_no_learn(self):
        """ Test no learning parameters """
        x = Variable(torch.Tensor(1, 10).fill_(2))
        dx = Variable(torch.Tensor(1, 10).fill_(2))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model, zero_mean=True,
                                            learn_mean=False, learn_rho=True)
        for p in var_model.parameters():
            self.assertTrue(p.grad is None)
        out = var_model(x)
        out.backward(dx)
        for p in var_model.parameters():
            self.assertTrue((p.grad.abs().sum() > 0).data[0])

        x = Variable(torch.Tensor(1, 10).fill_(2))
        dx = Variable(torch.Tensor(1, 10).fill_(2))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model, zero_mean=True,
                                            learn_mean=True, learn_rho=False)
        for p in var_model.parameters():
            self.assertTrue(p.grad is None)
        out = var_model(x)
        out.backward(dx)
        for p in var_model.parameters():
            self.assertTrue((p.grad.abs().sum() > 0).data[0])

        x = Variable(torch.Tensor(1, 10).fill_(2))
        dx = Variable(torch.Tensor(1, 10).fill_(2))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model, zero_mean=False,
                                            learn_mean=True, learn_rho=False)
        for p in var_model.parameters():
            self.assertTrue(p.grad is None)
        out = var_model(x)
        out.backward(dx)
        for p in var_model.parameters():
            self.assertTrue((p.grad.abs().sum() > 0).data[0])

    def test_mixtgauss(self):
        """Test mixt gauss prior"""
        x = Variable(torch.Tensor(1, 10).fill_(2))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model)
        var_model.set_prior('mixtgauss', n_mc_samples=2, sigma_1=1/2**1, 
                            sigma_2=1/2**6, pi=1/2)
        var_model(x)
        prior_loss = var_model.prior_loss()
        prior_loss.backward()
        
    def test_conj(self):
        """ Test conjugate prior """
        x = Variable(torch.Tensor(1, 10).fill_(2))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model)
        var_model.set_prior('conjugate', n_mc_samples=2, alpha_0=.5, 
                            beta_0=.5, mu_0=.5, kappa_0=.5)
        var_model(x)
        prior_loss = var_model.prior_loss()
        prior_loss.backward()

    def test_conjknownmean(self):
        """Test conjugate prior with known mean"""
        x = Variable(torch.Tensor(1, 10).fill_(2))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model)
        var_model.set_prior('conjugate_known_mean', n_mc_samples=2, 
                            alpha_0=.5, beta_0=.5, mean=0.)
        var_model(x)
        prior_loss = var_model.prior_loss()
        prior_loss.backward()
        
        
