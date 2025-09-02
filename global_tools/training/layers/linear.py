import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from global_tools.training.functions import tsslbp, bptt
import numpy as np
from global_tools.training.layers.utils import *


class LinearLayer(nn.Linear):
    def __init__(self, network_config, config, name, in_shape, mute=False):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.layer_config = config
        self.network_config = network_config
        self.name = name
        self.type = config['type']
        self.in_shape = in_shape
        self.out_shape = [out_features]
        self.in_spikes = None
        self.out_spikes = None
        if 'neuron' in config:
            self.neuron_out = config['neuron']
        else:
            self.neuron_out = 'current'

        if 'weight_scale' in config:
            weight_scale = config['weight_scale']
        else:
            weight_scale = 1

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)

        if len(self.in_shape) == 3:
            self.flatten = True
        else:
            self.flatten = False

        nn.init.kaiming_normal_(self.weight)
        self.weight = torch.nn.Parameter(weight_scale * self.weight.cuda(), requires_grad=True)

        if not mute:
            if self.flatten:
                print('(flatten)', ':', self.in_shape, '-', [np.cumprod(self.in_shape)[-1]])
                print(name, ':', [np.cumprod(self.in_shape)[-1]], '-', self.out_shape, self.neuron_out)
                print("-----------------------------------------")
            else:
                print(name, ':', self.in_shape, '-', self.out_shape, self.neuron_out)
                print("-----------------------------------------")

        if 'bn' in config.keys() and config['bn']:
            self.bn = nn.BatchNorm1d(out_features).cuda()
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], x.shape[1], -1).contiguous()
        x = merge_temporal_dim(x)
        x = f.linear(x, self.weight, self.bias)
        x = self.bn(x)
        x = expand_temporal_dim(x, self.network_config['n_steps'])
        return x

    def forward_temporal(self, x):
        y = self.forward(x)
        if self.neuron_out in ['psp', 'spike']:
            y = tsslbp.TSSLBP.apply(y, self.network_config, self.layer_config)

        return y

    def forward_bptt(self, x):
        y = self.forward(x)
        if self.neuron_out in ['psp', 'spike']:
            neuron = bptt.BPTTModule()
            y = neuron(y, self.network_config, self.layer_config)
        return y

    def forward_relu(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1).contiguous()
        x = f.linear(x, self.weight, self.bias)
        x = self.bn(x)
        if self.neuron_out == 'no_activation':
            return x
        else:
            return f.relu(x)

    def forward_softplus(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1).contiguous()
        x = f.linear(x, self.weight, self.bias)
        x = self.bn(x)
        if self.neuron_out == 'no_activation':
            return x
        else:
            return f.softplus(x)

    def forward_rnn(self, x):
        x = f.linear(x, self.weight, self.bias)
        x = self.bn(x)
        return x

    # def forward_spkprop(self, x):
    #     y = self.forward(x)
    #     if self.neuron_out in ['psp', 'spike']:
    #         neuron = spkprop.SPKPROPmodule()
    #         y = neuron(y, self.network_config, self.layer_config)
    #     return y

    def get_parameters(self):
        l = [self.weight]
        if 'bn' in self.layer_config.keys() and self.layer_config['bn']:
            l.append(self.bn.weight)
            l.append(self.bn.bias)
        return l

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w
