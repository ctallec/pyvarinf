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

def rebuild_parameters(dico, module, epsilon_setting):
    """ Rebuild parameters.
    
    Build the computational graph corresponding to
    the computations of the parameters of the given module,
    using the corresponding variational parameters in dico,
    and the rule used to sample epsilons. If the module has
    submodules, corresponding subcomputational graphs are also
    built.

    Typically, if a module has a parameter weight, weight should
    appear in dico, and the parameter will be rebuilt as
    module.weight = dico['weight'].mean + (1+dico['weight'].rho.exp()).log() *\
            dico['weight'].eps

    :args dico: a 'tree' dictionnary that contains variational
    parameters for the current module, and subtrees for submodules
    :args module: the module whose parameters are to be rebuilt
    :args epsilon_settings: how epsilons ought to be drawn

    """

    for name, p in dico.items():
        if isinstance(p, VariationalParameter):
            if p.eps is None:
                dico[name] = p._replace(eps=Variable(p.mean.data.clone()))
            epsilon_setting(name, dico[name])
            setattr(module, name, dico[name].mean +\
                    (1+p.rho.exp()).log() * dico[name].eps)
        elif p is None:
            setattr(module, name, None)
        else:
            rebuild_parameters(p, getattr(module, name), epsilon_setting)

def prior_std(p):
    """ Compute a reasonable prior standard deviation for parameter p.

    :args p: the parameter

    :return: the resulting std

    """
    stdv = 1
    if p.dim() > 1:
        for i in range(p.dim()-1):
            stdv = stdv * p.size()[i+1]
        stdv = 1 / math.sqrt(stdv)
    else:
        stdv = 1e-2
    return stdv

def sub_prior_loss(dico):
    """ Compute the KL divergence between prior and parameters for
    all Variational Parameters in the tree dictionary dico.

    :args dico: tree dictionary
    :return: KL divergence between prior and current
    """
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
    """ Build a Variational model over the model given as input.
    
    Variationalize changes all parameters of the given model
    to allow learning of a gaussian distribution over the
    parameters using Variational inference. For more information, 
    see e.g. TODO: REF.

    :args model: the model on which VI is to be performed
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self.dico = OrderedDict()
        self.variationalize_module(self.dico, self.model, '')

    def variationalize_module(self, dico, module, prefix):
        to_erase = []
        for name, p in module._parameters.items():
            if p is None:
                dico[name] = None
            else:
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
        epsilon_setting = \
                lambda name, p: p.eps.data.normal_() if self.training else\
                lambda name, p: p.eps.data.zero_()
        rebuild_parameters(self.dico, self.model, epsilon_setting)
        return self.model(*input)

    def prior_loss(self):
        return sub_prior_loss(self.dico)

class Sample(nn.Module):
    """ Utility to sample a single model from a Variational Model.

    Sample is a decorator that wraps a variational model, sample
    a model from the current parameter distribution and make the
    model usable as any other pytorch model. The sample can be
    redrawn using the draw() method. Draw needs to be called
    once before the model can be used.

    :args var_model: Variational model from which the sample models
    are to be drawn
    """
    def __init__(self, var_model):
        super().__init__()
        self.var_model = var_model

        self.association = []

    def draw(self, association=None, var_dico=None):
        if association is None:
            self.association = []
            association = self.association
            var_dico = self.var_model.dico

        for name, p in var_dico.items():
            if isinstance(p, VariationalParameter):
                if p.eps is None:
                    var_dico[name] = p._replace(eps=Variable(p.mean.data.clone()))
                association.append((var_dico[name].eps, var_dico[name].eps.data.clone().normal_()))
            else:
                self.draw(association, p)

    def forward(self, *input):
        for p, drawn_value_p in self.association:
            p.data.copy_(drawn_value_p)
        epsilon_setting = \
                lambda name, p: 1
        rebuild_parameters(self.var_model.dico, self.var_model.model, epsilon_setting)
        return self.var_model.model(*input)
