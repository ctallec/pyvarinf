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

def rebuild_parameters(dico, module):
    for name, p in dico.items():
        if isinstance(p, VariationalParameter):
            p.eps.data.normal_()
            setattr(module, name, p.mean +\
                    (1+p.rho.exp()).log() * p.eps)
        else:
            rebuild_parameters(p, getattr(module, name))

class Variationalize(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self.dico = OrderedDict()
        self.variationalize_module(self.dico, self.model, '')

    def variationalize_module(self, dico, module, prefix):
        to_erase = []
        for name, p in module._parameters.items():
            stdv = 0
            if p.dim() > 1:
                for i in range(p.dim()-1):
                    stdv = stdv + p.size()[i+1]
                stdv = 1 / math.sqrt(stdv)
                init_rho = math.log(math.exp(stdv) - 1)
            else:
                init_rho = 1

            dico[name] = VariationalParameter(
                Parameter(p.data.clone().fill_(0)),
                Parameter(p.data.clone().fill_(init_rho)),
                Variable(p.data.clone()))
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
        rebuild_parameters(self.dico, self.model)
        return self.model(*input)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        nn.Conv2d
        self.fc1 = nn.Linear(10, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        input = self.fc3(input)
        return F.log_softmax(input)

model = Model()
var_model = Variationalize(model)
print(var_model(Variable(torch.Tensor(10, 10))))
for p in var_model.parameters():
    print(p)
