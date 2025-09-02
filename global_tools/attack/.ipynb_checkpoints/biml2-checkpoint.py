import torch
import torch.nn as nn

from global_tools.attack.pgdl2 import PGDL2

class BIML2(PGDL2):
    r"""
    altered from torchattack
    """
    def __init__(self, model, forward_function=None, eps=0.3,
                 alpha=2/255, steps=40, T=None, **kwargs):
        super().__init__(model, forward_function=forward_function, eps=eps,
                 alpha=alpha, steps=steps, random_start=False, T=T, **kwargs)

