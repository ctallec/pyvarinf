from unittest import TestCase

import pyvarinf
import torch
import torch.nn as nn

from torch.autograd import Variable

class TestSample(TestCase):
    def test_sample_diff(self):
        x = Variable(torch.Tensor(1, 10).fill_(1))
        model = nn.Linear(10, 10)
        var_model = pyvarinf.Variationalize(model)
        sampled_model = pyvarinf.Sample(var_model)
        sampled_model.draw()
        a = sampled_model(x)
        b = sampled_model(x)
        self.assertTrue(a.eq(b).float().mean().data[0] == 1)
        sampled_model.draw()
        b = sampled_model(x)
        self.assertTrue(a.eq(b).float().norm().data[0] == 0)
