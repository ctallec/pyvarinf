import math
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class VILinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(VILinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mean_weight = Parameter(torch.zeros(out_features, in_features))
        self.rho_weight = Parameter(torch.Tensor(out_features, in_features))
        self.eps_weight = Variable(torch.Tensor(out_features, in_features))
        if bias:
            self.eps_bias = Variable(torch.Tensor(out_features))
            self.mean_bias = Parameter(torch.zeros(out_features))
            self.rho_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('mean_bias', None)
            self.register_parameter('rho_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        log_stdv = math.log(1. / math.sqrt(self.mean_weight.size(1)))
        self.rho_weight.data.fill_(log_stdv)
        if self.mean_bias is not None:
            self.rho_bias.data.fill_(0)

    def forward(self, input):

        self.eps_weight.data.normal_()
        weight = self.mean_weight + (1 + self.rho_weight.exp()).log() *\
                self.eps_weight
        if self.mean_bias is None:
            return self._backend.Linear()(input, weight)
        else:
            self.eps_bias.data.normal_()
            bias = self.mean_bias + (1 +
                                     self.rho_bias.exp()).log()*self.eps_bias
            return self._backend.Linear()(input, weight, bias)

    def cuda(self):
        super().cuda()
        self.eps_weight.cuda()
        if self.mean_bias:
            self.eps_bias.cuda()

    def kl_loss(self):
        stdv = 1. / math.sqrt(self.mean_weight.size(1))
        sigma_weight = (1 + self.rho_weight.exp()).log()

        loss_weight = -(sigma_weight/stdv).log() +\
                (sigma_weight.pow(2) + self.mean_weight.pow(2))/2/stdv/stdv - 1/2

        stdv_bias = 1
        if self.mean_bias is not None:
            sigma_bias = (1 + self.rho_bias.exp()).log()
            loss_bias = -(sigma_bias/stdv_bias).log() +\
                     (sigma_bias.pow(2) +
                      self.mean_bias.pow(2))/2/stdv_bias/stdv_bias - 1/2

        return loss_weight.sum() + loss_bias.sum()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

