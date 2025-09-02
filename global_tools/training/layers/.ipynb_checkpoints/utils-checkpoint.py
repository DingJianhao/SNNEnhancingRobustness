import torch
import torch.nn as nn


def merge_temporal_dim(x_seq):
    return x_seq.flatten(0, 1).contiguous()

def expand_temporal_dim(x_seq, T):
    y_shape = [T, int(x_seq.shape[0]/T)]
    y_shape.extend(x_seq.shape[1:])
    return x_seq.view(y_shape)