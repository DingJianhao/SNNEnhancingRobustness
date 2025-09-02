import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch


def get_fashionmnist(data_path, network_config):
    print("loading FashionMNIST")
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    batch_size = network_config['batch_size']
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

