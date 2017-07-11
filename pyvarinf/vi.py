import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn.parameter import Parameter

VariationalParameter = namedtuple('VariationalParameter',
                                  ['mean', 'rho', 'eps'])

def rebuild_parameters(dico, module, train):
    for name, p in dico.items():
        if isinstance(p, VariationalParameter):
            if p.eps is None:
                p = p._replace(eps=Variable(p.mean.data.clone()))
            if train:
                p.eps.data.normal_()
            else:
                p.eps.data.fill_(0)
            setattr(module, name, p.mean +\
                    (1+p.rho.exp()).log() * p.eps)
        else:
            rebuild_parameters(p, getattr(module, name), train)

def prior_std(p):
    stdv = 1
    if p.dim() > 1:
        for i in range(p.dim()-1):
            stdv = stdv * p.size()[i+1]
        stdv = 1 / math.sqrt(stdv)
    else:
        stdv = 1e-2
    return stdv

def sub_prior_loss(dico):
    loss = 0
    for name, p in dico.items():
        if isinstance(p, VariationalParameter):
            mean = p.mean
            std = (1+p.rho.exp()).log()
            std_prior = prior_std(mean)
            loss += (-(std/std_prior).log() + (std.pow(2) +\
                         mean.pow(2)) / (2 * std_prior ** 2) - 1/2).sum()
        else:
            loss += sub_prior_loss(p)
    return loss

class Variationalize(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self.dico = OrderedDict()
        self.variationalize_module(self.dico, self.model, '')

    def variationalize_module(self, dico, module, prefix):
        to_erase = []
        for name, p in module._parameters.items():
            stdv = prior_std(p)
            init_rho = math.log(math.exp(stdv) - 1)

            dico[name] = VariationalParameter(
                Parameter(p.data.clone().fill_(0)),
                Parameter(p.data.clone().fill_(init_rho)),
                None)

            self.register_parameter(prefix + '.' + name + '_mean',
                                   dico[name].mean)
            self.register_parameter(prefix + '.' + name + '_rho',
                                   dico[name].rho)


            to_erase.append(name)

        for name in to_erase:
            delattr(module, name)

        for mname, sub_module in module.named_children():
            sub_dico = OrderedDict()
            self.variationalize_module(sub_dico, sub_module, 
                                       prefix + ('.' if prefix else '')+\
                                      mname)
            dico[mname] = sub_dico

    def forward(self, *input):
        rebuild_parameters(self.dico, self.model, self.training)
        return self.model(*input)

    def prior_loss(self):
        return sub_prior_loss(self.dico)
