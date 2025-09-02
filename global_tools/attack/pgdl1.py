import torch
import torch.nn as nn
from global_tools.attack.attack import Attack
from torch.distributions import laplace
from torch.distributions import uniform
import numpy as np

def normalize_by_pnorm(x, p=1):
    batch_size = x.size(0)
    norm = x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(p)
    norm = torch.max(norm, torch.ones_like(norm) * 1e-6)
    x = (x.transpose(0, -1) * (1. / norm)).transpose(0, -1).contiguous()
    return x

def batch_l1_proj_flat(x, z=1):
    """
    Implementation of L1 ball projection from:

    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    from:

    https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246

    :param x: input data
    :param eps: l1 radius

    :return: tensor containing the projection.
    """

    # Computing the l1 norm of v
    v = torch.abs(x)
    v = v.sum(dim=1)

    # Getting the elements to project in the batch
    indexes_b = torch.nonzero(v > z).view(-1)
    if isinstance(z, torch.Tensor):
        z = z[indexes_b][:, None]
    x_b = x[indexes_b]
    batch_size_b = x_b.size(0)

    # If all elements are in the l1-ball, return x
    if batch_size_b == 0:
        return x

    # make the projection on l1 ball for elements outside the ball
    view = x_b
    view_size = view.size(1)
    mu = view.abs().sort(1, descending=True)[0]
    vv = torch.arange(view_size).float().to(x.device)
    st = (mu.cumsum(1) - z) / (vv + 1)
    u = (mu - st) > 0
    if u.dtype.__str__() == "torch.bool":  # after and including torch 1.2
        rho = (~u).cumsum(dim=1).eq(0).sum(1) - 1
    else:  # before and including torch 1.1
        rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
    theta = st.gather(1, rho.unsqueeze(1))
    # proj_x_b = _thresh_by_magnitude(theta, x_b)
    proj_x_b = torch.relu(torch.abs(x_b) - theta) * x_b.sign()

    # gather all the projected batch
    proj_x = x.detach().clone()
    proj_x[indexes_b] = proj_x_b
    return proj_x

def l1_proj(x, eps):
    batch_size = x.size(0)
    view = x.view(batch_size, -1)
    proj_flat = batch_l1_proj_flat(view, z=eps)
    return proj_flat.view_as(x)

class PGDL1(Attack):
    r"""
    structure altered from torchattack, inspired from advertorch
    """
    def __init__(self, model, forward_function=None,
                 eps=1.0, alpha=0.2, steps=40, random_start=True, eps_for_division=1e-10,sparsity=0.5,
                 T=None, **kwargs):
        super().__init__("PGDL2", model)
        self.eps = eps
        self.sparsity = sparsity
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self._supported_mode = ['default', 'targeted']
        self.forward_function = forward_function
        self.T = T
        self.kwargs = kwargs

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        delta = torch.zeros_like(adv_images).to(self.device)
        if self.random_start:
            # Starting at a uniformly random point
            ini = laplace.Laplace(
                loc=delta.new_tensor(0), scale=delta.new_tensor(1)
            )
            delta = ini.sample(delta.shape)

            delta = normalize_by_pnorm(delta, p=1)
            ray = uniform.Uniform(0, self.eps).sample()
            delta *= ray
            delta = torch.clamp(images + delta, 0, 1) - images

        for _ in range(self.steps):
            adv_images.requires_grad = True
            if self.forward_function is not None:
                outputs = self.forward_function(self.model, adv_images, self.T, **self.kwargs)
            else:
                outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            abs_grad = torch.abs(grad)
            batch_size = grad.size(0)
            view = abs_grad.view(batch_size, -1)
            view_size = view.size(1)
            if self.sparsity is None:
                vals, idx = view.topk(1)
            else:
                vals, idx = view.topk(
                    int(np.round((1 - self.sparsity) * view_size)))
            out = torch.zeros_like(view).scatter_(1, idx, vals)
            out = out.view_as(grad)
            grad = grad.sign() * (out > 0).float()
            grad = normalize_by_pnorm(grad, p=1)
            delta = delta + self.alpha * grad
            
            delta = l1_proj(delta, self.eps)
            delta = delta.to(self.device)
            delta = torch.clamp(images + delta, 0, 1) - images
            
            adv_images = adv_images.detach() + delta
        adv_images = torch.clamp(images + delta, 0, 1)
        return adv_images
