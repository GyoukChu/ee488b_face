#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
# Densenet201
def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.densenet201(num_classes=nOut)
