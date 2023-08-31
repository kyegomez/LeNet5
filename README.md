[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Paper-Implementation-Template
A simple implementation of LeNet5 for practice for the book "Pytorch Pocket Reference"

LeNet is abunch of convolution and linear layers with max pools.

Paper Link

# Appreciation
* Lucidrains
* Agorians


# Install
`pip install lenet5`

# Usage
```python
import torch
from lenet5 import LeNet5

x = torch.randn(1, 3, 32, 32)

model = LeNet5()

result = model(x)
print(result)
print(result.shape)
print(result.dtype)
```

# Architecture
The `LeNet5` architecture is composed of:

=> 2 convolutional layers with varying inc hannels and kernel sizes.

=> Linear layers

=> 2x max pool2d layers applied on relu -> convolutional layers(x) 

=> view, resizes according to -1 and the int of the element of the first dimension

=> 2 relus on the linear layers respectively

=> final linear projection

# License
MIT

# Citation

GradientBased Learning Applied to Document
Recognition
Yann LeCun Leon Bottou Yoshua Bengio and Patrick Hafner
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf