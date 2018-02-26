# pylint: disable=too-many-arguments, too-many-locals
""" Variational inference """
import math
from collections import namedtuple
from collections import OrderedDict
from scipy.special import gammaln
import numpy as np

import torch.nn as nn
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
            setattr(module, name, dico[name].mean +
                    (1 + p.rho.exp()).log() * dico[name].eps)
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
    for p in dico.values():
        if isinstance(p, VariationalParameter):
            mean = p.mean
            std = (1+p.rho.exp()).log()
            std_prior = prior_std(mean)
            loss += (-(std/std_prior).log() +
                     (std.pow(2) + mean.pow(2)) /
                     (2 * std_prior ** 2) - 1/2).sum()
        else:
            loss += sub_prior_loss(p)
    return loss

def sub_conjprior_loss(dico, module, alpha_0, beta_0, mu_0, kappa_0):
    """ Compute an estimation of the KL divergence between the conjugate
    prior and parameters for all Variational Parameters in the tree
    dictionary dico.

    :args dico: tree dictionary
    :args module: the module whose KL will be computed
    :args alpha_0: hyperparameter of the conjugate prior
    :args beta_0: hyperparameter of the conjugate prior
    :args mu_0: hyperparameter of the conjugate prior
    :args kappa_0: hyperparameter of the conjugate prior
    :return: estimation of the KL divergence between prior and current
    """
    logprior = 0.
    for name, p in dico.items():
        if isinstance(p, VariationalParameter):
            theta = getattr(module, name)
            S = (theta.mean() - mu_0).norm() ** 2
            V = (theta - theta.mean()).norm() ** 2
            n = np.prod(theta.size())
            alpha_n = alpha_0 + n/2
            kappa_n = kappa_0+n
            beta_n = beta_0 + V/2 + S * (kappa_0*n)/(2 * kappa_n)
            logprior -= - beta_n.log() * alpha_n + alpha_0 * np.log(beta_0) + \
                gammaln(alpha_n) - gammaln(alpha_0) + \
                .5 * np.log(kappa_0/kappa_n) - .5 * n * np.log(2*np.pi)
            std = (1+p.rho.exp()).log()
            H = std.log().sum() + .5 * n * (1 + np.log(2*np.pi))
            logprior -= H
        else:
            logprior += sub_conjprior_loss(\
                p, getattr(module, name), alpha_0, beta_0, mu_0, kappa_0)
    return logprior


def sub_conjpriorknownmean_loss(dico, module, mean, alpha_0, beta_0):
    """ Compute an estimation of the KL divergence between the conjugate
    prior when the mean is known and parameters for all Variational
    Parameters in the tree dictionary dico.

    :args dico: tree dictionary
    :args module: the module whose KL will be computed
    :args mean: known mean for the conjugate prior
    :args alpha_0: hyperparameter of the conjugate prior
    :args beta_0: hyperparameter of the conjugate prior
    :args mu_0: hyperparameter of the conjugate prior
    :args kappa_0: hyperparameter of the conjugate prior
    :return: estimation of the KL divergence between prior and current
    """
    logprior = 0.
    for name, p in dico.items():
        if isinstance(p, VariationalParameter):
            theta = getattr(module, name)
            S = (theta - mean).norm() ** 2
            n = np.prod(theta.size())
            alpha_n = alpha_0 + n/2
            beta_n = beta_0 + S/2
            logprior -= - beta_n.log() * alpha_n + \
                gammaln(alpha_n) - gammaln(alpha_0) + \
                alpha_0 * np.log(beta_0) - .5 * n * np.log(2*np.pi)
            std = (1+p.rho.exp()).log()
            H = std.log().sum() + .5 * n * (1 + np.log(2*np.pi))
            logprior -= H
        else:
            logprior += sub_conjpriorknownmean_loss(\
                p, getattr(module, name), mean, alpha_0, beta_0)
    return logprior


class Variationalize(nn.Module):
    """ Build a Variational model over the model given as input.

    Variationalize changes all parameters of the given model
    to allow learning of a gaussian distribution over the
    parameters using Variational inference. For more information,
    see e.g. TODO: REF.

    :args model: the model on which VI is to be performed
    :args zero_mean: if True, sets initial mean to 0, else
        keep model initial mean
    :args learn_mean: if True, learn the posterior mean
    :args learn_rho: if True, learn the posterior rho
    """
    def __init__(self, model, zero_mean=True, learn_mean=True, learn_rho=True):
        super().__init__()
        self.model = model

        self.dico = OrderedDict()
        self._variationalize_module(self.dico, self.model, '', zero_mean,
                                    learn_mean, learn_rho)

    def _variationalize_module(self, dico, module, prefix, zero_mean,
                               learn_mean, learn_rho):
        to_erase = []
        for name, p in module._parameters.items():  # pylint: disable=protected-access, line-too-long
            if p is None:
                dico[name] = None
            else:
                stdv = prior_std(p)
                init_rho = math.log(math.exp(stdv) - 1)

                init_mean = p.data.clone()
                if zero_mean:
                    init_mean.fill_(0)

                dico[name] = VariationalParameter(
                    Parameter(init_mean),
                    Parameter(p.data.clone().fill_(init_rho)),
                    None)

                if learn_mean:
                    self.register_parameter(prefix + '.' + name + '_mean',
                                            dico[name].mean)
                if learn_rho:
                    self.register_parameter(prefix + '.' + name + '_rho',
                                            dico[name].rho)

            to_erase.append(name)

        for name in to_erase:
            delattr(module, name)

        for mname, sub_module in module.named_children():
            sub_dico = OrderedDict()
            self._variationalize_module(sub_dico, sub_module,
                                        prefix + ('.' if prefix else '') +
                                        mname, zero_mean,
                                        learn_mean, learn_rho)
            dico[mname] = sub_dico

    def forward(self, *inputs):
        def _epsilon_setting(name, p):  # pylint: disable=unused-argument
            if self.training:
                return p.eps.data.normal_()
            return p.eps.data.zero_()

        rebuild_parameters(self.dico, self.model, _epsilon_setting)
        return self.model(*inputs)

    def prior_loss(self):
        """ Returns the prior loss """
        return sub_prior_loss(self.dico)

    def conjprior_loss(self, alpha_0=.5, beta_0=.5, mu_0=0, kappa_0=1.):
        return sub_conjprior_loss(self.dico, self.model, alpha_0,
                                  beta_0, mu_0, kappa_0)

    def conjpriorknownmean_loss(self, mean=0., alpha_0=.5, beta_0=.5):
        return sub_conjpriorknownmean_loss(self.dico, self.model, mean,
                                           alpha_0, beta_0)



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
        """ Draw a single model from the posterior variationally learned """
        if association is None:
            self.association = []
            association = self.association
            var_dico = self.var_model.dico

        for name, p in var_dico.items():
            if isinstance(p, VariationalParameter):
                if p.eps is None:
                    var_dico[name] = p._replace(eps=Variable(
                        p.mean.data.clone()))
                association.append((var_dico[name].eps,
                                    var_dico[name].eps.data.clone().normal_()))
            else:
                self.draw(association, p)

    def forward(self, *inputs):
        for p, drawn_value_p in self.association:
            p.data.copy_(drawn_value_p)

        def _epsilon_setting(name, p):  # pylint: disable=unused-argument
            return 1
        rebuild_parameters(self.var_model.dico, self.var_model.model,
                           _epsilon_setting)
        return self.var_model.model(*inputs)
