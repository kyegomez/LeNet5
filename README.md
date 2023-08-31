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

# Todo


# License

# Citations

