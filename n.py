import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# class Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # flatten
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28*28, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x


# load fashion mnist dataset
images = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# IMAGE PROCESSING
# Create a 512 x 512 matrix with random values
# img = np.random.random((112, 112)).astype(np.float32) * 255
# img_tensor = torch.from_numpy(img)
img_tensor = images[0][0]

# MODEL INSTANCE AND FORWARD PASS
model = Model()
# pass the image through the model
output = model(img_tensor)
print(output)

# plot the image with matplotlib
plt.imshow(img_tensor.detach().numpy().squeeze(), cmap='gray')
plt.show()




