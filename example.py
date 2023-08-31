
import torch
from lenet5 import LeNet5

x = torch.randint(0, 10, 10)

model = LeNet5()

model(x)
