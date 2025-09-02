import torch
import torch.nn as nn
import torch.nn.functional as f
from time import time
import global_tools.training.global_v as glv
from global_tools.training.layers.utils import *


class TSSLBP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config):
        shape = inputs.shape
        n_steps = network_config['n_steps']
        theta_m = 1 / network_config['tau_m']
        tau_s = network_config['tau_s']
        if layer_config['neuron'] == 'spike':
            tau_s = 1.
        theta_s = 1 / tau_s
        threshold = layer_config['threshold']
        if 'soft' in network_config.keys():
            soft = network_config['soft']
        else:
            soft = False

        mem = 0
        syn = 0
        syns_posts = []
        mems = []
        mem_updates = []
        outputs = []
        for t in range(n_steps):
            mem_update = (-theta_m) * mem + inputs[t]
            mem += mem_update

            out = mem > threshold
            out = out.type(torch.float32)

            mems.append(mem)

            if soft:
                mem = mem - out * threshold
            else:
                mem = mem * (1 - out)
            outputs.append(out)
            mem_updates.append(mem_update)
            syn = syn + (out - syn) * theta_s
            syns_posts.append(syn)

        mems = torch.stack(mems, dim=0)
        mem_updates = torch.stack(mem_updates, dim=0)
        outputs = torch.stack(outputs, dim=0)
        syns_posts = torch.stack(syns_posts, dim=0)
        ctx.save_for_backward(mem_updates, outputs, mems, syns_posts, torch.tensor([threshold, tau_s, theta_m]))
        return syns_posts

    @staticmethod
    def backward(ctx, grad_delta):
        (delta_u, outputs, u, syns, others) = ctx.saved_tensors
        shape = grad_delta.shape
        n_steps = shape[0]
        threshold = others[0].item()
        tau_s = others[1].item()
        theta_m = others[2].item()

        th = 1 / (4 * tau_s)
        grad = torch.zeros_like(grad_delta)

        if len(shape) == 5:
            syn_a = glv.syn_a.repeat(1, shape[1], shape[2], shape[3], shape[4])
            partial_a = glv.syn_a / (-tau_s)
            partial_a = partial_a.repeat(1, shape[1], shape[2], shape[3], shape[4])
        else:
            syn_a = glv.syn_a.reshape(-1,1,1).repeat(1, shape[1], shape[2])
            partial_a = glv.syn_a.reshape(-1,1,1) / (-tau_s)
            partial_a = partial_a.repeat(1, shape[1], shape[2])

        o = 0 #torch.zeros(shape[1], shape[2], shape[3], shape[4]).cuda()
        theta = 0 #torch.zeros(shape[1], shape[2], shape[3], shape[4]).cuda()

        for t in range(n_steps - 1, -1, -1):
            time_end = n_steps
            time_len = time_end - t
            out = outputs[t]
            partial_u = torch.clamp(-1 / delta_u[t], -8, 0) * out
            # current time is t_m
            # partial_a_partial_u = partial_u.unsqueeze(0).repeat(time_len, 1, 1, 1, 1) * partial_a[0:time_len]
            partial_a_partial_u = partial_u.unsqueeze(0) * partial_a[0:time_len]
            grad_tmp = torch.sum(partial_a_partial_u * grad_delta[t:time_end] * tau_s, dim=0)
            if t != n_steps - 1:
                grad_tmp += theta * u[t] * (-1) * theta_m * partial_u
                grad_tmp += theta * (1 - theta_m) * (1 - out)
            # current time is t_p
            theta = grad_tmp * out + theta * (1 - out) * (1 - theta_m)
            grad_a = torch.sum(syn_a[0:time_len] * grad_delta[t:time_end], dim=0)
            a = 0.2
            f = torch.clamp((-1 * u[t] + threshold) / a, -8, 8)
            f = torch.exp(f)
            f = f / ((1 + f) * (1 + f) * a)
            grad_a = grad_a * f
            syn = syns[t]
            grad_tmp[syn < th] = grad_a[syn < th]
            grad[t] = grad_tmp
        return grad, None, None
