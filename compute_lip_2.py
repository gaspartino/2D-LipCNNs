import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from model import *

def lipschitz_cnn(layer, input_size, iter = 500):

    weight = layer.weight
    stride = layer.stride
    padding = layer.padding

    x = torch.randn(input_size)
    for _ in range(iter):

        x = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
        x = F.conv_transpose2d(x, weight, bias=None, stride=stride, padding=padding)
        x = F.normalize(x, dim=(1, 2, 3))

    x = F.conv2d(x,weight,bias=None,stride=stride,padding=padding)
    return x.norm()

def lipschitz_cnn_2(layer, input_size, iter = 500):

    weight = layer.weight
    stride = layer.stride
    padding = layer.padding

    x = torch.randn(input_size)
    y = layer(x)
    for _ in range(iter):

        y = F.conv_transpose2d(y, weight, bias=None, stride=stride, padding=padding)
        y = F.conv2d(y, weight, bias=None, stride=stride, padding=padding)
        y = F.normalize(y, dim=(1, 2, 3))

    y = F.conv_transpose2d(y, weight, bias=None, stride=stride, padding=padding)
    return y.norm()

def lipschitz_fc(layer, input_size, iter=500):

    weight = layer.weight

    x = torch.randn(input_size)
    x = F.normalize(x)

    for _ in range(iter):
        x = F.linear(x, weight)
        x = F.linear(x, weight.T)
        x = F.normalize(x)

    x = F.linear(x, weight)
    return x.norm()

def lipschitz_upper_bound(model):
    lipschitz_constant = 1.0

    if isinstance(model, Vanilla2C2F):
        input_size = (1, 1, 32, 32)
        lipschitz_constant *= lipschitz_cnn(model.model[0], input_size)
        input_size = (1, 16, 16, 16)
        lipschitz_constant *= lipschitz_cnn(model.model[2], input_size)
        input_size = (1, 32*8*8)
        lipschitz_constant *= lipschitz_fc(model.model[5], input_size)
        input_size = (1, 100)
        lipschitz_constant *= lipschitz_fc(model.model[7], input_size)
    elif isinstance(model, Vanilla2C2FPool):
        input_size = (1, 1, 32, 32)
        lipschitz_constant *= lipschitz_cnn(model.conv1, input_size)
        input_size = (1, 16, 16, 16)
        lipschitz_constant *= lipschitz_cnn(model.conv2, input_size)
        input_size = (1, 32*8*8)
        lipschitz_constant *= lipschitz_fc(model.fc1, input_size)
        input_size = (1, 100)
        lipschitz_constant *= lipschitz_fc(model.fc2, input_size)
    elif isinstance(model, FCModel):
        input_size = (1, 32*32)
        lipschitz_constant *= lipschitz_fc(model.FC1, input_size)
        #input_size = (1, 64)
        #lipschitz_constant *= lipschitz_fc(model.FC2, input_size)
  
    #for layer in model.model:
    #    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #        input_size = x.size() if isinstance(x, torch.Tensor) else None
    #        lipschitz_constant *= generic_power_method(layer, input_size)
        
    
    return lipschitz_constant