import torch
import torch.nn as nn
import numpy as np
from global_tools.attack.attack import Attack

def project_L0_box(y, k, lb, ub):
    ''' projection of the batch y to a batch x such that:
            - each image of the batch x has at most k pixels with non-zero channels
            - lb <= x <= ub '''
    temp = y.clone()
    y = y.detach().permute(0, 2, 3, 1).cpu().numpy()
    lb = lb.detach().permute(0, 2, 3, 1).cpu().numpy()
    ub= ub.detach().permute(0, 2, 3, 1).cpu().numpy()

    x = np.copy(y)
    p1 = np.sum(x**2, axis=-1)
    p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
    p2 = np.sum(p2**2, axis=-1)
    p3 = np.sort(np.reshape(p1-p2, [p2.shape[0],-1]))[:,-k]
    x = x*(np.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
    x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
    
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(temp)
    return x

class PGDL0(Attack):
    def __init__(self, model, forward_function=None, eps=500, k=5, steps=40, kappa=0.5, random_start=True, T=None, **kwargs):
        super().__init__("PGD", model)
        self.eps = eps
        self.kappa = kappa
        self.k = k
        self.steps = steps
        self.random_start = random_start
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

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.kappa, self.kappa)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            if i > 0:
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
                grad /= (1e-10 + torch.sum(torch.abs(grad), dim=(1,2,3)).reshape(-1,1,1,1))
                adv_images = adv_images.detach() + (torch.rand_like(grad).to(grad) - 0.5) * 1e-12 + self.eps * grad
            adv_images = images + project_L0_box(adv_images - images, self.k, -1 * images, 1.0 - images).detach()
        return adv_images
