# PyVarInf
PyVarInf provides facilities to easily train your PyTorch neural network models using variational inference.

# Bayesian Deep Learning with Variational Inference

## Bayesian Deep Learning
Assume we have a dataset D = {(x<sub>1</sub>, y<sub>1</sub>), ...,
(x<sub>n</sub>, y<sub>n</sub>)} where the x's are the inputs and the y's the
outputs. The problem is to predict the y's from the x's. Further assume that
p(D|θ) is the output of a neural network with *weights* θ. The *network loss*
is defined as

<img src="https://latex.codecogs.com/gif.latex?L^n(\theta)&space;=&space;-\log&space;p(D|\theta)&space;=&space;-\sum_i&space;\log&space;p(y_i|x_i,&space;\theta)" title="L^n(\theta) = -\log p(D|\theta) = -\sum_i \log p(y_i|x_i, \theta)" />

Usually, when training a neural network, we try to find the parameter θ* which minimizes L<sup>n</sup>(θ).

In Bayesian Inference, the problem is instead to study the posterior
distribution of the weights given the data. Assume we have a prior α over
ℝ<sup>d</sup>. The posterior is

<img
src="https://latex.codecogs.com/gif.latex?p(\theta|D)&space;=&space;\frac{p(D|\theta)\alpha(\theta)}{\int_\theta&space;p(D|\theta)\alpha(\theta)&space;d\theta}"
title="p(\theta|D) = \frac{p(D|\theta)\alpha(\theta)}{\int_\theta
p(D|\theta)\alpha(\theta) d\theta}" />

This can be used for model selection, or prediction with Bayesian Model Averaging.

## Variational Inference
It is usually impossible to analytically compute the posterior distribution,
especially with models as complex as neural networks. Variational Inference adress
this problem by approximating the posterior p(θ|D) by a parametric distribution
q(θ|φ) where φ is a parameter. The problem is then not to learn a parameter θ*
but a probability distribution q(θ|φ) minimizing 

<img src="https://latex.codecogs.com/gif.latex?F(\phi)&space;=&space;\mathbf{E}_{\theta&space;\sim&space;q(\theta|\phi)}[L^N(\theta)]&space;&plus;&space;KL(q(.|\phi)\|\alpha)" title="F(\phi) = E_{\theta \sim q(\theta|\phi)}[L^N(\theta)] + KL(q(.|\phi)\|\alpha)" />

F is called the *variational free energy*.

This idea was originally introduced for deep learning by Hinton and Van Camp
[5] as a way to use neural networks for Minimum Description Length [3].
MDL aims at minimizing the number of bits used to encode the whole dataset.
Variational inference introduces one of many data encoding schemes.
Indeed, F can be interpreted as the total description length of the dataset D,
when we first encode the model, then encode the part of the data not explained by
the model: 
* L<sup>C</sup>(φ) = KL(q(.|φ)||α) is the complexity loss. It measures (in
  nats) the quantity of information contained in the model. It is indeed
  possible to encode the model in L<sup>C</sup>(φ) nats, with the *bits-back*
  code [4].
* L<sup>E</sup>(φ) = __E__<sub>θ ~ q(θ|φ)</sub>[L<sup>n</sup>(θ)] is the error
  loss. It measures the necessary quantity of information for encoding the data
  D with the model. This code length can be achieved with a Shannon-Huffman
  code for instance.

Therefore F(φ) = L<sup>C</sup>(φ) + L<sup>E</sup>(φ) can be rephrased as an
MDL loss function which measures the total encoding length of the data.

## Practical Variational Optimisation
In practice, we define φ = (µ, σ) in ℝ<sup>d</sup> x ℝ<sup>d</sup>, and q(.|φ)
= N(µ, Σ) the multivariate distribution where Σ =
diag(σ<sub>1</sub><sup>2</sup>, ..., σ<sub>d</sub><sup>2</sup>), and we want to
find the optimal µ* and σ*.

With this choice of a gaussian posterior, a Monte Carlo estimate of the
gradient of F w.r.t. µ and σ can be obtained with backpropagation. This allows
to use any gradient descent method used for non-variational optimisation [2]


# Overview of PyVarInf
The core feature of PyVarInf is the `Variationalize` function. `Variationalize`
takes a model as input and outputs a variationalized version of the model with
gaussian posterior.

## Definition of a variational model
To define a variational model, first define a traditional PyTorch model, then
use the Variationalize function : 
```python
import pyvarinf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)

    def forward(self, x):
        x = self.bn1(F.relu(F.max_pool2d(self.conv1(x), 2)))
        x = self.bn2(F.relu(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
var_model = pyvarinf.Variationalize(model)
var_model.cuda()
```

## Optimisation of a variational model
Then, the `var_model` can be trained that way : 
```python
optimizer = optim.Adam(var_model.parameters(), lr=0.01)

def train(epoch):
    var_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = var_model(data)
        loss_error = F.nll_loss(output, target)
	# The model is only sent once, thus the division by
	# the number of datapoints used to train
        loss_prior = var_model.prior_loss() / 60000
        loss = loss_error + loss_prior
        loss.backward()
        optimizer.step()

for epoch in range(1, 500):
    train(epoch)
```

## Available priors

In PyVarInf, we have implemented four families of priors : 

### Gaussian prior
The gaussian prior is N(0,Σ), with Σ the diagonal matrix diag(σ<sub>1</sub><sup>2</sup>, ..., σ<sub>d</sub><sup>2</sup>) defined such that 1/σ<sub>i</sub> is the square root of the number of parameters in the layer, following the standard initialisation of neural network weights. 
It is the default prior, and do not have any parameter. It can be set with : 
```python
var_model.set_prior('gaussian')
```
### Conjugate priors
The conjugate prior is used if we assume that all the weights in a given layer should be distributed as a gaussian, but with unknown mean and variance. See [6] for more details. This prior can be set with 
```python
var_model.set_prior('conjugate', n_mc_samples, alpha_0, beta_0, mu_0, kappa_0)
```
There are five parameters that have to bet set : 
- `n_mc_samples`, the number of samples used in the Monte Carlo estimation of the prior loss and its gradient.
* `mu_0`, the prior sample mean
* `kappa_0`, the number of samples used to estimate the prior sample mean
* `alpha_0` and `beta_0`, such that variance was estimated from 2 alpha_0 observations with sample mean mu_0 and sum of squared deviations 2 beta_0
             
### Conjugate prior with known mean
The conjugate prior with known mean is similar to the conjugate prior. It is used if we assume that all the weights in a given layer should be distributed as a gaussian with a known mean but unknown variance. It is usefull in neural networks model when we assume that the weights in a layer should have mean 0. See [6] for more details. This prior can be set with :
```python
var_model.set_prior('conjugate_known_mean', n_mc_samples, mean, alpha_0, beta_0)
```
Four parameters have to be set:
* `n_mc_samples`, the number of samples used in the Monte Carlo estimation of the prior loss and its gradient.
* `mean`, the known mean
* `alpha_0` and `beta_0` defined as above



### Mixture of two gaussian
The idea of using a mixture of two gaussians is defined in [1]. This prior can be set with: 
```python
var_model.set_prior('mixtgauss', n_mc_samples, sigma_1, sigma_2, pi)
```
* `n_mc_samples`, the number of samples used in the Monte Carlo estimation of the prior loss and its gradient.
* `sigma_1` and `sigma_2` the std of the two gaussians
* `pi` the probability of the first gaussian

# Requirements
This module requires Python 3. You need to have PyTorch installed for PyVarInf to work (as PyTorch is not readily available on PyPi). To install PyTorch, follow the instructions described [here](http://pytorch.org/#pip-install-pytorch).

# References
* [1] Blundell, Charles, Cornebise, Julien, Kavukcuoglu, Koray, and Wierstra, Daan. Weight Uncertainty in Neural Networks. In *International Conference on Machine Learning*, pp. 1613–1622, 2015.
* [2] Graves, Alex. Practical Variational Inference for Neural Networks. In *Neural Information Processing Systems*, 2011.
* [3] Grünwald, Peter D. *The Minimum Description Length principle*. MIT press, 2007.
* [4] Honkela, Antti and Valpola, Harri. Variational Learning and Bits-Back Coding: An Information-Theoretic View to Bayesian Learning. *IEEE transactions on Neural Networks*, 15(4), 2004.
* [5] Hinton, Geoffrey E and Van Camp, Drew. Keeping Neural Networks Simple by Minimizing the Description Length of the Weights. In *Proceedings of the sixth annual conference on Computational learning theory*. ACM, 1993.
* [6] Murphy, Kevin P. *Conjugate Bayesian analysis of the Gaussian distribution.*, 2007.
