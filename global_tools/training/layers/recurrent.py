import torch.nn.functional as f
import torch.nn.init as init
from global_tools.training.layers.utils import *
import numpy as np

class LSTM(nn.Module):
    def __init__(self, network_config, config, name, in_shape, mute=False):
        super(LSTM, self).__init__()
        self.layer_config = config
        self.network_config = network_config
        self.name = name
        self.type = config['type']
        self.in_shape = in_shape
        if len(self.in_shape) == 3:
            self.flatten = True

        if 'output' in config:
            self.neuron_out = config['output']
        else:
            self.neuron_out = 'last_output'
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layer = config['num_layer']
        if 'bidirectional' in config:
            self.bidirectional = config['bidirectional']
        else:
            self.bidirectional = True

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layer,
                            batch_first=False, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.out_shape = [self.hidden_size * 2]
        else:
            self.out_shape = [self.hidden_size]

        if not mute:
            if self.flatten:
                print('(flatten)', ':', self.in_shape, '-', [np.cumprod(self.in_shape)[-1]])
                print(name, ':', [np.cumprod(self.in_shape)[-1]], '-', self.out_shape, self.neuron_out)
                print("-----------------------------------------")
            else:
                print(name, ':', self.in_shape, '-', self.out_shape, self.neuron_out)
                print("-----------------------------------------")


    def forward_rnn(self, x):
        if len(x.shape) == 3: # T, B, N
            assert(x.shape[0] == self.network_config['n_steps'])
        elif len(x.shape) == 5: # T, B, C, H, W
            assert (x.shape[0] == self.network_config['n_steps'])

        batch_size = x.shape[1]
        x = x.reshape(self.network_config['n_steps'], batch_size, -1)
        if self.bidirectional:
            hidden_state = torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).to(x)
            cell_state = torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).to(x)
            output, (last_hidden_state, last_cell_state) = self.lstm(x, (hidden_state, cell_state))
        else:
            hidden_state = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(x)
            cell_state = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(x)
            output, (last_hidden_state, last_cell_state) = self.lstm(x, (hidden_state, cell_state))

        if self.neuron_out == 'last_output':
            return output[-1]
        elif self.neuron_out == 'mean_output':
            return output.mean(0)

    def weight_clipper(self):
        pass

class GRU(nn.Module):
    def __init__(self, network_config, config, name, in_shape, mute=False):
        super(GRU, self).__init__()
        self.layer_config = config
        self.network_config = network_config
        self.name = name
        self.type = config['type']
        self.in_shape = in_shape
        if len(self.in_shape) == 3:
            self.flatten = True

        if 'output' in config:
            self.neuron_out = config['output']
        else:
            self.neuron_out = 'last_output'
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_layer = config['num_layer']
        if 'bidirectional' in config:
            self.bidirectional = config['bidirectional']
        else:
            self.bidirectional = True

        self.gru = nn.GRU(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layer,
                            batch_first=False, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.out_shape = [self.hidden_size * 2]
        else:
            self.out_shape = [self.hidden_size]

        if not mute:
            if self.flatten:
                print('(flatten)', ':', self.in_shape, '-', [np.cumprod(self.in_shape)[-1]])
                print(name, ':', [np.cumprod(self.in_shape)[-1]], '-', self.out_shape, self.neuron_out)
                print("-----------------------------------------")
            else:
                print(name, ':', self.in_shape, '-', self.out_shape, self.neuron_out)
                print("-----------------------------------------")


    def forward_rnn(self, x):
        if len(x.shape) == 3: # T, B, N
            assert(x.shape[0] == self.network_config['n_steps'])
        elif len(x.shape) == 5: # T, B, C, H, W
            assert (x.shape[0] == self.network_config['n_steps'])

        batch_size = x.shape[1]
        x = x.reshape(self.network_config['n_steps'], batch_size, -1)
        if self.bidirectional:
            hidden_state = torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).to(x)
            output, last_hidden_state = self.gru(x, hidden_state)
        else:
            hidden_state = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(x)
            output, last_hidden_state = self.gru(x, hidden_state)

        if self.neuron_out == 'last_output':
            return output[-1]
        elif self.neuron_out == 'mean_output':
            return output.mean(0)

    def weight_clipper(self):
        pass