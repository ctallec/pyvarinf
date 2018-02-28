# PyVarInf
PyVarInf provides facilities to easily train your PyTorch neural network models using variational inference.

# Overview
The core feature of PyVarInf is the `Variationalize` function. `Variationalize` takes a model as input and outputs a variationalized version of the model with gaussian posterior.

Variationalized models are similar to traditional PyTorch models, except that they resample their parameters according to their current posterior each time forward is called. They also provide two helper functions `set_prior` and `prior_loss`. `set_prior` allows the user to specify the prior to be used (gaussian by default). `prior_loss` yields the loss term associated to the selected prior, i.e. an approximation of KL(q || prior), where q is the current posterior.

# Requirements
You need to have PyTorch installed for PyVarInf to work (as PyTorch is not readily available on PyPi). To install PyTorch, follow the instructions described [here](http://pytorch.org/#pip-install-pytorch).
