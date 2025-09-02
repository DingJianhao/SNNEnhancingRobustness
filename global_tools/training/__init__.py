import yaml
import torch

class parse(object):
    """
    This class reads yaml parameter file and allows dictionary like access to the members.
    """
    def __init__(self, path):
        with open(path, 'r') as file:
            self.parameters = yaml.safe_load(file)

    # Allow dictionary like access
    def __getitem__(self, key):
        return self.parameters[key]

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)

class ZNORM(torch.nn.Module):
    def __init__(self, mean=None, std=None):
        super(ZNORM, self).__init__()
        if mean is not None and std is not None:
            self.mean = torch.Tensor(mean).reshape(1, -1, 1, 1).cuda()
            self.std = torch.Tensor(std).reshape(1, -1, 1, 1).cuda()
        else:
            self.mean, self.std = None, None

    def forward(self, x):
        # x batch, C, W, H
        if len(x.shape) == 4 and self.mean is not None and self.std is not None:
            return (x - self.mean) / self.std
        else:
            return x