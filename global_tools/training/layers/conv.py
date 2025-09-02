import torch.nn.functional as f
import torch.nn.init as init
from global_tools.training.functions import tsslbp, bptt
from global_tools.training.layers.utils import *

class ConvLayer(nn.Conv2d):
    def __init__(self, network_config, config, name, in_shape, groups=1, mute=False):
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        self.type = config['type']
        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        if 'neuron' in config:
            self.neuron_out = config['neuron']
        else:
            self.neuron_out = 'current'


        if 'padding' in config:
            padding = config['padding']
        else:
            padding = 0

        if 'stride' in config:
            stride = config['stride']
        else:
            stride = 1

        if 'dilation' in config:
            dilation = config['dilation']
        else:
            dilation = 1

        if 'weight_scale' in config:
            weight_scale = config['weight_scale']
        else:
            weight_scale = 1

        kernel = kernel_size

        super(ConvLayer, self).__init__(in_features, out_features, kernel, stride, padding, dilation, groups,
                                        bias=False)
        # nn.init.kaiming_normal_(self.weight)
        self.weight = torch.nn.Parameter(weight_scale * self.weight.cuda(), requires_grad=True)

        self.in_shape = in_shape
        self.out_shape = [out_features, int((in_shape[1]+2*padding-kernel)/stride+1),
                          int((in_shape[2]+2*padding-kernel)/stride+1)]
        if not mute:
            print(self.name, ':', self.in_shape, '-', self.out_shape, self.neuron_out)
            print("-----------------------------------------")

        if 'bn' in config.keys() and config['bn']:
            self.bn = nn.BatchNorm2d(out_features).cuda()
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        x = merge_temporal_dim(x)
        x = f.conv2d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
        x = self.bn(x)
        x = expand_temporal_dim(x, self.network_config['n_steps'])
        return x

    def get_parameters(self):
        l = [self.weight]
        if 'bn' in self.layer_config.keys() and self.layer_config['bn']:
            l.append(self.bn.weight)
            l.append(self.bn.bias)
        return l

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
        x = f.conv2d(x, self.weight, self.bias,
                     self.stride, self.padding, self.dilation, self.groups)
        x = self.bn(x)
        return f.relu(x)

    def forward_softplus(self, x):
        x = f.conv2d(x, self.weight, self.bias,
                     self.stride, self.padding, self.dilation, self.groups)
        x = self.bn(x)
        return f.softplus(x)

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w
