import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from global_tools.training.layers.utils import *
from global_configs import enc_func

class MixLayer(nn.Module):
    def __init__(self, network_config, config, name, mute=False):
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        self.type = config['type']

        if not mute:
            print(self.name, ':', ','.join(self.layer_config['enc']))
            print("-----------------------------------------")

        super().__init__()

    def forward(self, x):
        T = self.network_config['n_steps']
        N = len(self.layer_config['enc'])
        C = x.shape[1]
        inputs_t = torch.zeros(T, x.shape[0], x.shape[1] * N,
                                       x.shape[2], x.shape[3]).cuda()
        for k, enc in enumerate(self.layer_config['enc']):
            for t in range(T):
                inputs_t[t,:,k * C: (k+1) * C,:,:] = self.znorm(enc_func(x, t, T, mode=enc))
        return inputs_t

    def forward_pass(self, x):
        y1 = self.forward(x)
        return y1

    def weight_clipper(self):
        return

    def get_parameters(self):
        return