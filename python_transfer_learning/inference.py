# -*- coding: utf-8 -*-
import torchvision
from torch import optim


def load_model():
    model_conv = torchvision.models.mobilenet_v2(pretrained=False)
    optimizer = optim.SGD(*args, **kwargs)
    model_conv.load_state_dict()