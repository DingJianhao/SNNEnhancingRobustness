import torch
import torch.nn.functional as f
import global_tools.training.global_v as glv


def psp(inputs, network_config):
    shape = inputs.shape
    n_steps = network_config['n_steps']
    tau_s = network_config['tau_s']

    syn = torch.zeros(shape[1], shape[2], shape[3], shape[4]).cuda()
    syns = torch.zeros(n_steps, shape[1], shape[2], shape[3], shape[4]).cuda()

    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[t, ...]
        syns[t, ...] = syn / tau_s

    return syns


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """
    def __init__(self, network_config):
        super(SpikeLoss, self).__init__()
        self.network_config = network_config
        self.criterion = torch.nn.CrossEntropyLoss()

    def spike_count(self, outputs, target, network_config, layer_config):
        delta = loss_count.apply(outputs, target, network_config, layer_config)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_latency(self, outputs, target, network_config, beta=2):
        T = outputs.shape[0]
        decreasing_output = outputs * torch.arange(T, 0, -1).view(-1, 1, 1).to(outputs)
        t_min = T + 1 - torch.max(decreasing_output, dim=0).values
        prob = torch.softmax(-1 * beta * t_min, dim=1)
        return f.cross_entropy(prob, target)
    
    def spike_latency2(self, outputs, target, network_config):
        T = outputs.shape[0]
        decreasing_output = outputs * torch.arange(T, 0, -1).view(-1, 1, 1).to(outputs)
        dec = torch.max(decreasing_output, dim=0).values
        return f.cross_entropy(dec, target)
    
    def spike_latency3(self, outputs, target, network_config):
        T = outputs.shape[0]
        decreasing_output = outputs * torch.arange(T, 0, -1).view(-1, 1, 1).to(outputs)
        dec = torch.max(decreasing_output, dim=0).values
        return f.cross_entropy(2 * (dec - T), target)


class loss_count(torch.autograd.Function):  # a and u is the incremnet of each time steps
    @staticmethod
    def forward(ctx, outputs, target, network_config, layer_config):
        desired_count = network_config['desired_count']
        undesired_count = network_config['undesired_count']
        shape = outputs.shape
        n_steps = shape[0]
        out_count = torch.sum(outputs, dim=0)

        delta = (out_count - target) / n_steps
        mask = torch.ones_like(out_count)
        mask[target == undesired_count] = 0
        mask[delta < 0] = 0
        delta[mask == 1] = 0
        mask = torch.ones_like(out_count)
        mask[target == desired_count] = 0
        mask[delta > 0] = 0
        delta[mask == 1] = 0
        delta = delta.unsqueeze_(0).repeat(n_steps, 1, 1)
        return delta

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None

