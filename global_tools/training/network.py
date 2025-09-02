import torch
import torch.nn as nn
import global_tools.training.layers.conv as conv
import global_tools.training.layers.pooling as pooling
import global_tools.training.layers.dropout as dropout
import global_tools.training.layers.linear as linear
import global_tools.training.layers.res_block as res_block
import global_tools.training.layers.recurrent as recurrent
import global_tools.training.layers.mix_encode as mix_encode
import global_tools.training.functions.loss_f as f
import global_tools.training.global_v as glv


class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape, mute=False):
        super(Network, self).__init__()
        self.layers = []
        self.network_config = network_config
        self.layers_config = layers_config
        parameters = []
        if not mute:
            print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if c['type'] == 'conv':
                self.layers.append(conv.ConvLayer(network_config, c, key, input_shape, mute=mute))
                input_shape = self.layers[-1].out_shape
                parameters.extend(self.layers[-1].get_parameters())
            elif c['type'] == 'res_block':
                self.layers.append(res_block.ResBlock(network_config, c, key, input_shape, mute=mute))
                input_shape = self.layers[-1].out_shape
                parameters.extend(self.layers[-1].get_parameters())
            elif c['type'] == 'linear':
                self.layers.append(linear.LinearLayer(network_config, c, key, input_shape, mute=mute))
                input_shape = self.layers[-1].out_shape
                parameters.extend(self.layers[-1].get_parameters())
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(network_config, c, key, input_shape, mute=mute))
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'lstm':
                self.layers.append(recurrent.LSTM(network_config, c, key, input_shape, mute=mute))
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'gru':
                self.layers.append(recurrent.GRU(network_config, c, key, input_shape, mute=mute))
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'dropout':
                self.layers.append(dropout.DropoutLayer(c, key))
            elif c['type'] == 'mix_enc':
                self.layers.append(mix_encode.MixLayer(network_config, c, key, mute=mute))
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))
        self.layers = nn.ModuleList(self.layers)
        # self.layers[0].mergeT = True

        # self.my_parameters = nn.ParameterList(parameters)
        # if not mute:
        #     print("-----------------------------------------")

    def forward(self, spike_input, is_train):
        if isinstance(self.network_config["encode"], list):
            spikes = spike_input
        elif not self.network_config["rule"] == "BP":
            spikes = f.psp(spike_input, self.network_config)
        else:
            spikes = spike_input

        for l in self.layers:
            if l.type == "dropout":
                if is_train:
                    spikes = l(spikes)
            elif l.type == 'mix_enc':
                spikes = l.forward_pass(spikes)
                spikes = f.psp(spikes, self.network_config)
            elif l.type == 'pooling':
                spikes = l.forward_pass(spikes)
            elif "TSSLBP" in self.network_config["rule"] and self.network_config['model'] == "LIF":
                spikes = l.forward_temporal(spikes)
            elif "BPTT" in self.network_config["rule"] and self.network_config['model'] == "LIF":
                spikes = l.forward_bptt(spikes)
            elif self.network_config["rule"] == "BP" and self.network_config["model"] == "ReLU":
                spikes = l.forward_relu(spikes)
            elif self.network_config["rule"] == "BP" and self.network_config["model"] == "Softplus":
                spikes = l.forward_softplus(spikes)
            elif self.network_config["rule"] == "BPTT" and self.network_config['model'] == "RNN":
                spikes = l.forward_rnn(spikes)
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(self.network_config['rule']))
            
            # if isinstance(l, res_block.ResBlock):
            #     print(spikes.sum())
        return spikes

    def get_parameters(self):
        return self.layers.parameters()

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def train(self, mode=True):
        for l in self.layers:
            l.train(mode)
        return self

    def eval(self):
        return self.train(False)
