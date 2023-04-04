"""
The idea of this script is to test several ideas about convolutions:
- The role of striding in the receptive field
- The role of dilated convolutions to increase the receptive field
- Check equivalences of striding and non striding according to Kong et al. (2017)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2
import numpy as np
from typing import List


# import the cifar dataset
cifar = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

# create the data loader
cifar_dataloader = DataLoader(cifar, batch_size=32, shuffle=True)

# create the model
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(10*28*28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = nn.Softmax(dim=1)(x)
        return x


# create loss and optimizer
model = BaseModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train the model
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        epoch_loss = 0
        for batch, (image, label) in enumerate(dataloader):
            # forward pass
            pred = model(image)
            loss = criterion(pred, label)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the loss
            epoch_loss += loss.item()
        
        # print the loss
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(dataloader)}")

train_model(model, cifar_dataloader, criterion, optimizer, epochs=5)


# plot with opencv
test_image = cifar[0][0]
relu = nn.ReLU()
with torch.no_grad():
    conv1_numpy = relu(model.conv1(test_image))
    conv2_numpy = relu(model.conv2(conv1_numpy)).detach().numpy()
    conv1_numpy = conv1_numpy.detach().numpy()

# function to extract each image and add a border
def extract_image(image: np.array):
    new_image = cv2.copyMakeBorder(
        image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value= 1
    )
    return new_image

def h_concat(conv_result: np.array):
    """
    conv_result: is the activation map of one convolutional layer
    """
    # extract each image, the firs dimesion is the number of filters, the second and third are the image
    list_images = [extract_image(image) for image in conv_result]
    # concatenate the images
    h_image = cv2.hconcat(list_images)
    return h_image

def v_concat_after_h_concat(conv_result: List[np.array]):
    """
    conv_result: is the list of activation map of each convolutional layer in numpy format
    """
    # h concate the images
    # breakpoint()
    h_concat_images = [h_concat(image) for image in conv_result]
    # get the width of each image and extract the max
    width = max([image.shape[1] for image in h_concat_images])
    # resize each of the images with the max width
    resized_images = [cv2.resize(image, (width, image.shape[0])) for image in h_concat_images]
    # v concat the images
    v_concat_images = cv2.vconcat(resized_images)
    return v_concat_images

# concatenate the images
cv_image = v_concat_after_h_concat([conv1_numpy, conv2_numpy])

# plot the image
cv2.imshow("conv1", cv_image)
cv2.waitKey(0)

# TODO compute accuracy
# TODO create a validation dataset
# TODO train more epochs











