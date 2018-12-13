# pylint: disable=too-many-arguments, too-many-locals
""" Variational inference """
import math
import functools
from collections import namedtuple
from collections import OrderedDict
from scipy.special import gammaln
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

VariationalParameter = namedtuple('VariationalParameter',
                                  ['mean', 'rho', 'eps'])


def evaluate(variational_parameter):
    """ Evaluates the current value of a variational parameter.

    Returns mean + log(1 + e^rho) * eps
    :args variational_parameter: the variational parameter
    :returns: the value of the variational parameter
    """
    assert isinstance(variational_parameter, VariationalParameter), \
        "Incorrect type."
    return variational_parameter.mean + \
        (1 + variational_parameter.rho.exp()).log() * variational_parameter.eps


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
    module.weight = dico['weight'].mean + (1+dico['weight'].rho.exp()).log() *
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
            setattr(module, name, evaluate(dico[name]))
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
        for i in range(p.dim() - 1):
            stdv = stdv * p.size()[i + 1]
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
            std = (1 + p.rho.exp()).log()
            std_prior = prior_std(mean)
            loss += (-(std / std_prior).log() +
                     (std.pow(2) + mean.pow(2)) /
                     (2 * std_prior ** 2) - 1 / 2).sum()
        else:
            loss += sub_prior_loss(p)
    return loss


def sub_entropy(dico):
    """ Compute the entropy of the parameters for all Variational
    Parameters in the tree dictionary dico.
    :args dico: tree dictionary
    :returns: Entropy of the current distribution
    """
    entropy = 0.
    for _, p in dico.items():
        if isinstance(p, VariationalParameter):
            std = (1 + p.rho.exp()).log()
            n = np.prod(std.size())
            entropy += std.log().sum() + .5 * n * (1 + np.log(2 * np.pi))
        else:
            entropy += sub_entropy(p)
    return entropy


def sub_conjprior(dico, alpha_0, beta_0, mu_0, kappa_0):
    """ Compute an estimation of the KL divergence between the conjugate
    prior and parameters for all Variational Parameters in the tree
    dictionary dico.

    :args dico: tree dictionary
    :args alpha_0: hyperparameter of the conjugate prior
    :args beta_0: hyperparameter of the conjugate prior
    :args mu_0: hyperparameter of the conjugate prior
    :args kappa_0: hyperparameter of the conjugate prior
    :return: estimation of the KL divergence between prior and current
    """
    logprior = 0.
    for _, p in dico.items():
        if isinstance(p, VariationalParameter):
            theta = evaluate(p)
            S = (theta.mean() - mu_0).norm() ** 2
            V = (theta - theta.mean()).norm() ** 2
            n = np.prod(theta.size())
            alpha_n = alpha_0 + n / 2
            kappa_n = kappa_0 + n
            beta_n = beta_0 + V / 2 + S * (kappa_0 * n) / (2 * kappa_n)
            logprior += - beta_n.log() * alpha_n + alpha_0 * np.log(beta_0) + \
                gammaln(alpha_n) - gammaln(alpha_0) + \
                .5 * np.log(kappa_0 / kappa_n) - .5 * n * np.log(2 * np.pi)

        else:
            logprior += sub_conjprior(
                p, alpha_0, beta_0, mu_0, kappa_0)
    return logprior


def sub_conjpriorknownmean(dico, mean, alpha_0, beta_0):
    """ Compute an estimation of the KL divergence between the conjugate
    prior when the mean is known and parameters for all Variational
    Parameters in the tree dictionary dico.

    :args dico: tree dictionary
    :args mean: known mean for the conjugate prior
    :args alpha_0: hyperparameter of the conjugate prior
    :args beta_0: hyperparameter of the conjugate prior
    :return: estimation of the KL divergence between prior and current
    """
    logprior = 0.
    for _, p in dico.items():
        if isinstance(p, VariationalParameter):
            theta = evaluate(p)
            S = (theta - mean).norm() ** 2
            n = np.prod(theta.size())
            alpha_n = alpha_0 + n / 2
            beta_n = beta_0 + S / 2
            logprior += - beta_n.log() * alpha_n + \
                gammaln(alpha_n) - gammaln(alpha_0) + \
                alpha_0 * np.log(beta_0) - .5 * n * np.log(2 * np.pi)
        else:
            logprior += sub_conjpriorknownmean(
                p, mean, alpha_0, beta_0)
    return logprior


def sub_mixtgaussprior(dico, sigma_1, sigma_2, pi):
    """ Compute an estimation of the KL divergence between the prior
    defined by the mixture of two gaussian distributions
    for all Variational Parameters in the tree dictionary dico.
    More details on this prior and the notations can be found in :
    "Weight Uncertainty in Neural Networks" Blundell et al, 2015
    https://arxiv.org/pdf/1505.05424.pdf

    :args dico: tree dictionary
    :args sigma_1: std of the first gaussian in the mixture
    :args sigma_2: std of the second gaussian in the mixture
    :args pi: probability of the first gaussian in the mixture
    :return: estimation of the KL divergence between prior and current
    """
    logprior = 0.
    for _, p in dico.items():
        if isinstance(p, VariationalParameter):
            theta = evaluate(p)
            n = np.prod(theta.size())
            theta2 = theta ** 2
            pgauss1 = (- theta2 / (2. * sigma_1 ** 2)).exp() / sigma_1
            pgauss2 = (- theta2 / (2. * sigma_2 ** 2)).exp() / sigma_2
            logprior += (pi * pgauss1 + (1 - pi) * pgauss2 + 1e-8).log().sum()
            logprior -= n / 2 * np.log(2 * np.pi)
        else:
            logprior += sub_mixtgaussprior(
                p, sigma_1, sigma_2, pi)
    return logprior


class Variationalize(nn.Module):
    """ Build a Variational model over the model given as input.

    Variationalize changes all parameters of the given model
    to allow learning of a gaussian distribution over the
    parameters using Variational inference. For more information,
    see e.g. https://papers.nips.cc/paper/4329-practical-variational
    -inference-for-neural-networks.pdf.

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
        self._prior_loss_function = functools.partial(
            sub_prior_loss,
            dico=self.dico)

    def _variationalize_module(self, dico, module, prefix, zero_mean,
                               learn_mean, learn_rho):
        to_erase = []
        paras = module._parameters.items()  # pylint: disable=protected-access
        for name, p in paras:
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
                    self.register_parameter(prefix + '_' + name + '_mean',
                                            dico[name].mean)
                if learn_rho:
                    self.register_parameter(prefix + '_' + name + '_rho',
                                            dico[name].rho)

            to_erase.append(name)

        for name in to_erase:
            delattr(module, name)

        for mname, sub_module in module.named_children():
            sub_dico = OrderedDict()
            self._variationalize_module(sub_dico, sub_module,
                                        prefix + ('_' if prefix else '') +
                                        mname, zero_mean,
                                        learn_mean, learn_rho)
            dico[mname] = sub_dico

    def set_prior(self, prior_type, **prior_parameters):
        """ Change the prior to be used.

        Available priors are 'gaussian', 'conjugate', 'mixtgauss' and
        'conjugate_known_mean'. For each prior, you must
        specify the corresponding parameter:
          - For the gaussian prior, no parameter is required.
          - For the conjugate prior, you must specify
            - n_mc_samples, the number of samples used in the Monte Carlo
              estimation of the prior loss and its gradient.
            - mu_0, the prior sample mean
            - kappa_0, the number of samples used to estimate the
              prior sample mean
            - alpha_0 and beta_0, such that variance was estimated from 2
             alpha_0 observations with sample mean mu_0 and sum of squared
             deviations 2 beta_0
          - For the conjugate prior with known mean,
            - n_mc_samples, the number of samples used in the Monte Carlo
              estimation of the prior loss and its gradient.
            - mean, the known mean
            - alpha_0 and beta_0 defined as above
          - For the mixture of two gaussians,
            - n_mc_samples, the number of samples used in the Monte Carlo
              estimation of the prior loss and its gradient.
            - sigma_1 and sigma_2 the std of the two gaussians
            - pi the probability of the first gaussian
        For further information, see:
            https://en.wikipedia.org/wiki/Conjugate_prior.
        Acts inplace by modifying the value of _prior_loss_function
        :args prior_type: one of 'gaussian', 'conjugate',
            'conjugate_known_mean', 'mixtgauss'
        :args prior_parameters: the parameters for the associated prior
        """
        if prior_type == 'gaussian':
            self._prior_loss_function = functools.partial(
                sub_prior_loss,
                dico=self.dico)
        else:
            n_mc_samples = prior_parameters.pop("n_mc_samples")
            if prior_type == 'conjugate':
                mc_logprior_function = functools.partial(
                    sub_conjprior,
                    **prior_parameters
                )
            if prior_type == 'conjugate_known_mean':
                mc_logprior_function = functools.partial(
                    sub_conjpriorknownmean,
                    **prior_parameters
                )
            if prior_type == 'mixtgauss':
                mc_logprior_function = functools.partial(
                    sub_mixtgaussprior,
                    **prior_parameters
                )

            def prior_loss_function():
                """Compute the prior loss"""
                logprior = 0.
                for _ in range(n_mc_samples):
                    rebuild_parameters(
                        self.dico, self.model,
                        lambda name, p: p.eps.data.normal_()
                    )
                    logprior += mc_logprior_function(self.dico)
                logprior = logprior / n_mc_samples
                H = sub_entropy(self.dico)
                prior_loss = - logprior - H
                return prior_loss
            self._prior_loss_function = prior_loss_function

    def forward(self, *inputs):
        def _epsilon_setting(name, p):  # pylint: disable=unused-argument
            if self.training:
                return p.eps.data.normal_()
            return p.eps.data.zero_()

        rebuild_parameters(self.dico, self.model, _epsilon_setting)
        return self.model(*inputs)

    def prior_loss(self):
        """ Returns the prior loss """
        return self._prior_loss_function()


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
