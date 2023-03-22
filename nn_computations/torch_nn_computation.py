"""
How the tensor are mutiplied in order to compute the activation layer?
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# load fashion mnist dataset
images = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# Flatten and linear layer
flatten = nn.Flatten()
linear = nn.Linear(28*28, 10)
softmax = nn.Softmax(dim=1)

# get first image and pass it through the model
img_tensor = images[0][0]
img_flatten = flatten(img_tensor)
img_linear = linear(img_flatten)
img_softmax = softmax(img_linear)

# print the result
print(img_softmax)

# ASSERTIONS
"""
Note the matrix multiplication in the below case for the assertion, the flatten result has a shape
of (1, 784) and the linear weight has a shape of (10, 784), the transpose is needed.
"""
m_linear = softmax(torch.matmul(linear.weight, img_flatten.transpose(0, 1)).transpose(1, 0) + linear.bias)
assert torch.allclose(img_softmax, m_linear)



