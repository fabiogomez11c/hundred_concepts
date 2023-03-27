"""
How is computed the activation layer when I have a convolution layer before?
What does it mean that the Convolution doesn't have all the activations connected between each
other, only connected with the small region that covers the filter?
"""
import torch
import numpy as np
import torch.nn as nn

image = torch.tensor(np.random.rand(3, 7, 7), dtype=torch.float32)
conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
activation = conv(image.unsqueeze(0))

"""
Basically it takes the first piece of the image (3, 3) and multiply it by the filter (3, 3) and then sum
all the elements of tha matrix (3,3) to give a scalar (1,) and we sum the bias (1,), that gives the result
of the first element of the activation layer, as we are flowing over the matrix, the activation layer
will be a matrix too. If we have several channels, the result stills being a scalar. On the other hand, if we have
several filters, that will be like the 'channel' of the activation layer (which can be interpreted as an 'image'), in
other words if we have 3 filters, we will have 3 matrices of activation layer stacked. (See notes page 10 and 11).

Remember, the scalar operation is to get an specific element inside the resultant matrix, it doesn't mean that
the result is a scalar or just a vector.
"""

# This is the operation to get the first element of the activation layer
assert torch.allclose(activation[0, 0, 0, 0], (conv.weight[0] * image[:, 0:3, 0:3]).sum() + conv.bias[0])
# This is the operation to get the second element of the activation layer
assert torch.allclose(activation[0, 0, 0, 1], (conv.weight[0] * image[:, 0:3, 1:4]).sum() + conv.bias[0])

# This is a validation
assert (conv.weight[0] * image[:, 0:3, 0:3])[0][0][0] == conv.weight[0][0][0][0] * image[:, 0:3, 0:3][0][0][0]

"""
Now, I'm going to get the activation using numpy.
"""

np_image = np.array(
    [
        [1, 1, 1, 1],
        [0.5, 0.5, 0.5, 0.5],
        [0, 0, 0, 0],
        [-1, -1, -1, -1]
    ]
)

np_weights = np.array(
    [
        [0.5, 0.5],
        [0.5, 0.5]
    ]
)

np_bias = np.array([1.5])

np_activation = np.zeros((3, 3))  # start the activation matrix with zeros
for i in range(3):  # through columns
    for j in range(3):  # through rows
        image_piece = np_image[j:j+2, i:i+2]  # get the piece of the image that is going to get filtered
        np_activation[j, i] = (image_piece * np_weights).sum() + np_bias  # get the activation element

# now the same with torch to end up comparing results
torch_image = torch.from_numpy(np_image).unsqueeze(0)
torch_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
torch_conv.weight = nn.Parameter(torch.from_numpy(np_weights).unsqueeze(0).unsqueeze(0))
torch_conv.bias = nn.Parameter(torch.from_numpy(np_bias))

torch_activation = torch_conv(torch_image.unsqueeze(0))

assert np.all(torch_activation.detach().numpy() == np_activation)
