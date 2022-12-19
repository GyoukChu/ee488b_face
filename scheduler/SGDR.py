#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, lr, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

	lr_step = 'epoch'

	print('Initialised SGDR scheduler')

	return sche_fn, lr_step
