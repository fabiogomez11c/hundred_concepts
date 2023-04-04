import torch
import torch.nn as nn
from torchvision import datasets, transforms


cifar = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)





















