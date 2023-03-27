import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# class Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # flatten
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28 * 28, 10)

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

images_dataloader = DataLoader(images, batch_size=100, shuffle=True)


# train one batch function
def train_one_batch(data_batch, model, loss_fn, optimizer):
    model.train()
    # get the data and the labels
    data, labels = data_batch
    # pass the data through the model
    batch_output = model(data)
    # compute the loss
    loss = loss_fn(batch_output, labels)
    # backpropagation
    loss.backward()
    # update the weights
    optimizer.step()
    # reset the gradients
    optimizer.zero_grad()
    # return the loss
    return loss.item()


# MODEL INSTANCE AND FORWARD PASS
model = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# plot each K row in the weight matrix reshaped to the same image dimension
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
im_list = []
for i in range(10):
    im_list.append(ax[i // 5, i % 5].imshow(model.linear.weight[i].reshape(28, 28).detach().numpy(), animated=True, cmap="gray"))
    # set title of the image
    ax[i // 5, i % 5].set_title(labels_map[i])
plt.show()


# create the animation
for i in range(1000):
    # im_list[0].set_array(np.random.randn(28, 28))
    one_batch = next(iter(images_dataloader))
    loss_batch = train_one_batch(one_batch, model, loss_fn, optimizer)

    # update the plot of weights
    for j, im in enumerate(im_list):
        im.set_array(model.linear.weight[j].reshape(28, 28).detach().numpy())

    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f"{i} Loss: {loss_batch}")
