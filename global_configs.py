import torch
import yaml

dataset_path = {
    'mnist': 'C:/Users/HUAWEI/Desktop/snn_robust_code/data/mnist',
    'fashionmnist': 'C:/Users/HUAWEI/Desktop/snn_robust_code/data/fashionmnist',
    'cifar10': 'C:/Users/HUAWEI/Desktop/snn_robust_code/data/cifar10'
}

ckpt_part2_url = 'https://drive.google.com/file/d/19d3CO2kJxejUE5pNz6QhatfVtJRXXneR/view?usp=sharing'
ckpt_part3_url = 'https://drive.google.com/file/d/1u5KGqPOhrUKz1DMyLzMBoy-_A0zqAr3K/view?usp=sharing'
ckpt_part4_url = 'https://drive.google.com/file/d/1FbTAU1kz0elVn282_LPKsH-1ns-n5l0T/view?usp=sharing'
atkimg_part4_url = 'https://drive.google.com/file/d/1Zs-xbnAJpcLHIPs1t2Yy3N0xaX1Z0cCX/view?usp=sharing'

dataset_class_number = {
    'mnist': 10,
    'fashionmnist': 10,
    'cifar10': 10,
    'dvsgesture': 11
}

def enc_func(data, t, T, mode='poisson'):
    if mode.lower() == 'poisson':
        return torch.rand_like(data).le(data * 1.0).to(data).float()
    elif mode.lower() == 'const':
        return data
    elif mode.lower() == 'pre':
        return ((t / T) < data).float()
    elif mode.lower() == 'post':
        return 1.0 - ((t / T) <= (1 - data)).float()
    else: # mode.lower() == 'latency':
        mask = torch.floor((1.0 - data) * T)
        return (t == mask).float()


class parse(object):
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
        if len(x.shape) == 4 and self.mean is not None and self.std is not None:
            return (x - self.mean) / self.std
        else:
            return x

def static_enc_func(img, T, mode='const'):
    if mode.lower() == 'const':
        return img
    elif mode.lower() == 'quantize':
        return torch.round_(img * T) / T

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha
        return grad_x, None
    
class poisson(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.rand_like(x).to(x).le(x).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ttfs_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T, t):
        mask = torch.floor((1.0 - x) * T)
        out = (t == mask).float()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0]
        return grad_output * out, None, None
    
class post_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T, t):
        out = 1.0 - ((t / T) <= (1 - x)).float()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0]
        return grad_output * out, None, None
    
class pre_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, T, t):
        out = ((t / T) < x).float()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.saved_tensors[0]
        return grad_output * out, None, None

def enc_func_differential(data, t, T, mode='poisson'):
    if mode.lower() == 'poisson':
        return poisson.apply(data)
    elif mode.lower() == 'const':
        return data
    elif mode.lower() == 'pre':
        return pre_func.apply(data, T, t)
    elif mode.lower() == 'post':
        return post_func.apply(data, T, t)
    else: # mode.lower() == 'latency':
        return ttfs_func.apply(data, T, t)

def force_cpu():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.cpu()
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self.cpu()

if __name__ == "__main__":
    data = torch.tensor(0.5)
    data.requires_grad = True
    T = 20
    v = []
    for t in range(20):
        v.append(enc_func_differential(data, t, T, mode='pre'))
    print(torch.stack(v))
    loss = torch.mean(torch.stack(v)) * 5 - 1
    loss.backward()
    print(data.grad)

