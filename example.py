
import torch
from lenet5 import LeNet5

input_data = torch.randn(1, 3, 32, 32)#.to(device=device)  # 3 channels for color image

model = LeNet5()

result = model(input_data)
print(result)
print(result.shape)
print(result.dtype)


