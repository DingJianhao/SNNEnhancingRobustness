import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from global_tools.training.functions import tsslbp, bptt
from global_tools.training.layers.utils import *
from torchvision.models.resnet import BasicBlock, conv1x1


# conv_1:
#     type: "res_block"
#     in_channels: 3
#     out_channels: 15
#     kernel_size: 5
#     padding: 0
#     threshold: 1
#     neuron: 'psp'

class ResBlock(BasicBlock):
    def __init__(self, network_config, config, name, in_shape, mute=False):
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        if 'neuron' in config:
            self.neuron_out = config['neuron']
        else:
            self.neuron_out = 'current'
        if 'dilation' in config:
            dilation = config['dilation']
        else:
            dilation = 1

        if 'stride' in config:
            stride = config['stride']
        else:
            stride = 1

        if 'groups' in config:
            groups = config['groups']
        else:
            groups = 1

        in_channels = config['in_channels']
        out_channels = config['out_channels']

        if in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            downsample = None

        super().__init__(inplanes=in_channels, planes=out_channels, stride=stride, downsample=downsample,
            groups=groups, base_width=64, dilation=dilation, norm_layer=nn.BatchNorm2d)
        self.relu = None
        self.in_shape = in_shape
        h = in_shape[1]
        h = int((h + 2 * self.conv1.padding[0] - self.conv1.kernel_size[0]) / self.conv1.stride[0] + 1)
        h = int((h + 2 * self.conv2.padding[0] - self.conv2.kernel_size[0]) / self.conv2.stride[0] + 1)
        self.out_shape = [out_channels, h, h]

        if 'bptt' in network_config['rule'].lower():
            self.bptt_neuron1 = bptt.BPTTModule()
            self.bptt_neuron2 = bptt.BPTTModule()

        if not mute:
            print(self.name, ':', self.in_shape, '-', self.out_shape, self.neuron_out)
            print("-----------------------------------------")

    def forward_temporal(self, x):
        x = merge_temporal_dim(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = expand_temporal_dim(out, self.network_config['n_steps'])
        out = tsslbp.TSSLBP.apply(out, self.network_config, self.layer_config)
        out = merge_temporal_dim(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = expand_temporal_dim(out, self.network_config['n_steps'])
        out = tsslbp.TSSLBP.apply(out, self.network_config, self.layer_config)
        return out

    def forward_bptt(self, x):
        x = merge_temporal_dim(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = expand_temporal_dim(out, self.network_config['n_steps'])
        out = self.bptt_neuron1(out, self.network_config, self.layer_config)
        out = merge_temporal_dim(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = expand_temporal_dim(out, self.network_config['n_steps'])
        out = self.bptt_neuron2(out, self.network_config, self.layer_config)
        return out

    def forward_relu(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = f.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = f.relu(out)
        return out

    def forward_softplus(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = f.softplus(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = f.softplus(out)
        return out

    def get_parameters(self):
        l = [self.conv1.weight, self.conv2.weight,
             self.bn1.weight, self.bn1.bias,
             self.bn2.weight, self.bn2.bias,]
        if self.conv1.bias is not None:
            l.append(self.conv1.bias)
        if self.conv2.bias is not None:
            l.append(self.conv2.bias)
        if self.downsample is not None:
            l.append(self.downsample[0].weight)
            if self.downsample[0].bias is not None:
                l.append(self.downsample[0].bias)
            l.append(self.downsample[1].weight)
            l.append(self.downsample[1].bias)
        return l

    def weight_clipper(self):
        w = self.conv1.weight.data
        w = w.clamp(-4, 4)
        self.conv1.weight.data = w

        w = self.conv2.weight.data
        w = w.clamp(-4, 4)
        self.conv2.weight.data = w



if __name__ == '__main__':
    # from torchvision.models.resnet import resnet18
    # print(resnet18())
    # exit(0)

    network_config = {
        'n_steps': 10,
        'tau_m': 2,
        'tau_s': 2,
        'model': "LIF",
        'rule': 'bptt'
    }
    layer_config = {
        'in_channels': 3,
        'out_channels': 15,
        'stride': 3,
        'threshold': 1,
        'neuron': 'psp'
    }

    block = ResBlock(network_config, layer_config, 'xxx', [3, 11, 11], mute=False)
    print(block.forward_bptt(torch.rand(10, 7, 3, 11, 11)).shape)