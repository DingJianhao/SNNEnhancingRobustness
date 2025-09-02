import torch
import torch.nn as nn
from global_tools.attack.attack import Attack

class EOTFGSM(Attack):
    def __init__(self, model, forward_function=None, repeat=10, eps=0.007, T=None, **kwargs):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.forward_function = forward_function
        self.T = T
        self.repeat = repeat
        self.kwargs = kwargs

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images = images + self.eps / 2 * torch.randn_like(images).to(self.device)
        images = torch.clamp(images, min=0, max=1).detach()

        if self._targeted:
            target_labels = self._get_target_label(images, labels)


        loss = nn.CrossEntropyLoss()

        images.requires_grad = True

        grad = 0
        for _ in range(self.repeat):
            if self.forward_function is not None:
                outputs = self.forward_function(self.model, images, self.T, **self.kwargs)
            else:
                outputs = self.model(images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            # Update adversarial images
            grad += torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]
        grad /= self.repeat

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

