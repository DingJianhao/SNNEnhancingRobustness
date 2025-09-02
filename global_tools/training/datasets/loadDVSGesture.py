from torchvision.datasets import DatasetFolder
from typing import Callable, Optional
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.utils import _log_api_usage_once
from functools import partial

def load_spikingjelly_npz(file_name: str) -> np.ndarray:
    data = np.load(file_name)
    t, x, y, p = data['t'], data['x'], data['y'], data['p']
    t = t - t.min()
    return torch.from_numpy(np.stack((x, y, t, p), axis=1))


class DVSGesture(DatasetFolder):
    def __init__(
            self,
            root: str,
            train=True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.train = train
        if self.train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')

        super().__init__(root=root, loader=load_spikingjelly_npz, extensions=('.npz',), transform=transform,
                         target_transform=target_transform)
        self.H = 128
        self.W = 128

        
def voxelgrid_forward(events, T, H, W, downsample=1, sum=False, start_time=None, end_time=None, temporal_dim_first=False, scale=None):
    x, y, t, p, b = events.t()
    # print(torch.min(t) / 1e6, torch.max(t) / 1e6)
    if end_time is not None:
        idx = t < end_time * 1e6
        x = x[idx]
        y = y[idx]
        t = t[idx]
        p = p[idx]
        b = b[idx]
    if start_time is not None:
        idx = t > start_time * 1e6
        x = x[idx]
        y = y[idx]
        t = t[idx]
        p = p[idx]
        b = b[idx]

    if isinstance(downsample, int):
        x = torch.floor(x / downsample)
        y = torch.floor(y / downsample)
        H = int(H/downsample)
        W = int(W/downsample)

    B = 1 + int(torch.max(b).item())
    num_voxels = int(2 * T * H * W * B)
    vox = events.new_full([num_voxels, ], fill_value=0)

    normed_t = torch.zeros_like(t)
    for bi in range(B):
        max_ = t[b == bi].max()
        min_ = t[b == bi].min()
        normed_t[b == bi] = (t[b == bi] - min_) / (max_ - min_)

    quantized_t = torch.floor(normed_t * (T - 1))

    idx = x \
            + W * y \
            + W * H * quantized_t \
            + W * H * T * p \
            + W * H * T * 2 * b

    idx = torch.clip(idx.long(), min=0, max=num_voxels - 1)
    values = torch.ones_like(x).to(x)
    if sum:
        vox.put_(idx, values, accumulate=True)
    else:
        vox.put_(idx, values, accumulate=False)
        vox = torch.clip_(vox, 0, 1)
    vox = vox.view(-1, 2, T, H, W)
    if temporal_dim_first:
        vox = vox.permute(2, 0, 1, 3, 4)
    else:
        vox = vox.permute(0, 2, 1, 3, 4)
    if scale is None:
        return torch.Tensor(vox)
    else:
        return torch.Tensor(vox) / scale

def collate_events(data, T, H, W, downsample=1, sum=False, start_time=None, end_time=None, temporal_dim_first=False, scale=None):
    labels = []
    events = []
    for i, d in enumerate(data): # i=batch_index, d=(x,y,t,p)
        labels.append(d[1])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]), 1), dtype=np.float32)], 1) # (x, y, t, p, batch)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0)).float()
    labels = default_collate(labels)
    grid = voxelgrid_forward(events, T=T, H=H, W=W, 
                                       downsample=downsample, sum=sum, 
                                       start_time=start_time, end_time=end_time, 
                                       temporal_dim_first=temporal_dim_first, scale=scale)
    return grid, labels


class DVSGestureLoader(DataLoader):
    def __init__(self, T, H, W, downsample=1, sum=False, start_time=None, end_time=None, temporal_dim_first=False, scale=None, **kwargs):
        kwargs['collate_fn'] = partial(collate_events, T=T, H=H, W=W, 
                                       downsample=downsample, sum=sum, 
                                       start_time=start_time, end_time=end_time, 
                                       temporal_dim_first=temporal_dim_first, scale=scale)
        super().__init__(**kwargs)


def get_dvsgesture(data_path, network_config):
    print("loading DVSGesture")
    batch_size = network_config['batch_size']
    trainset = DVSGesture(data_path, train=True)
    testset = DVSGesture(data_path, train=False)
    trainloader = DVSGestureLoader(T=60, H=trainset.H, W=trainset.W, downsample=1, 
                                   sum=True, start_time=0, end_time=6, 
                                   temporal_dim_first=True, scale=None, dataset=trainset, shuffle=True, batch_size=batch_size, num_workers=4)
    testloader = DVSGestureLoader(T=60, H=trainset.H, W=trainset.W, downsample=1, 
                                   sum=True, start_time=0, end_time=6, 
                                   temporal_dim_first=True, scale=None, dataset=testset, shuffle=False, batch_size=batch_size, num_workers=4)
    return trainloader, testloader


def load_frames(file_name: str):
    data = np.load(file_name)
    data_max = np.max(data)
    return torch.from_numpy(data / data_max)


class DVSGestureFrames(DatasetFolder):
    def __init__(
            self,
            root: str,
            train=True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.train = train
        if self.train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'test')

        super().__init__(root=root, loader=load_frames, extensions=('.npy',), transform=transform,
                         target_transform=target_transform)
        self.H = 128
        self.W = 128

def get_dvsgestureframe(data_path, network_config):
    print("loading DVSGestureFrame")
    batch_size = network_config['batch_size']
    trainset = DVSGestureFrames(data_path, train=True)
    testset = DVSGestureFrames(data_path, train=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainloader, testloader

if __name__ == '__main__':
    import sys
    sys.path.append('../../../')
    from global_configs import dataset_path
    trainloader, testloader = get_dvsgestureframe(dataset_path['dvsgestureframe'], {'batch_size': 64})
    print(len(testloader.dataset))
    # for i in trainloader:
    #     print(i[0].shape)

    exit(0)
    import sys
    from tqdm import tqdm
    sys.path.append('../../../')
    from global_configs import dataset_path
    trainloader, testloader = get_dvsgesture(dataset_path['dvsgesture'], {'batch_size': 1})
    id = 0
    test = 'train'
    if test == 'test':
        loader = testloader
    else:
        loader = trainloader
    for i in tqdm(loader):
        data, label = i
        T = data.shape[0]
        if not os.path.exists('/home/data10T/dingjh/DVSGesture/seperate_frames/%s/%d' % (test, label.item())):
            os.makedirs('/home/data10T/dingjh/DVSGesture/seperate_frames/%s/%d' % (test, label.item()))
        for t in range(T):
            np.save('/home/data10T/dingjh/DVSGesture/seperate_frames/%s/%d/%d-%d.npy' % (test, label.item(), id, t), data[t, 0].numpy())
        id += 1

    exit(0)
    import matplotlib.pyplot as plt
    # os.chdir('../../../')
    import sys
    sys.path.append('../../../')
    from global_configs import dataset_path
    trainloader, testloader = get_dvsgesture(dataset_path['dvsgesture'], {'batch_size': 64})
    mean1 = 0
    mean2 = 0
    std1 = 0
    std2 = 0
    for i in trainloader:
        data = i[0]
        print(data.shape, torch.min(data), torch.max(data))
        mean1 += torch.mean(data[:,:,0]).item()
        mean2 += torch.mean(data[:,:,1]).item()

        std1 += torch.std(data[:,:,0]).item()
        std2 += torch.std(data[:,:,1]).item()
        print(data.shape)
        v = np.zeros((3, 128, 128))
        v[:2] = data[5,0].numpy()
        plt.imshow(v.transpose(1,2,0) * 250)
        plt.savefig('ss.png')
        exit(0)
        print(data.max(), data.min())
    print(
        mean1/len(trainloader), mean2/len(trainloader),
        std1/len(trainloader), std2/len(trainloader),
          )
    # 0.03241567519542418 0.03886394202709198 0.17692723870277405 0.1931223187007402