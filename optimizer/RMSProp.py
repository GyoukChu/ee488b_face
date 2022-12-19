#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):

	print('Initialised RMSProp optimizer')

	return torch.optim.RMSprop(parameters, lr = lr, alpha=0.99, weight_decay = weight_decay,
									momentum=0.9);
