import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #3 = in channels -> 6=out_channels, 5=kernel_size
        self.conv2 = nn.Conv2d(6, 16, 5) #6=in_chanels, => 16=out_channels, 5=kernel_size
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #in_feats = 16x5x5, out_feats=120
        self.fc2 = nn.Linear(120, 84) #in_feats=120, out_feats=84
        self.fc3 = nn.Linear(84, 10)#in_feats=84, out_feats=10

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) #applies a max_pool2d(input: tensor(minibatch, in_channels, iH, iW)) op on the relu on the conv and applies it on x
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #applies a max_pool2s op(input: tensor(minibatch, in_channels, iH, IW) op on the relu of the 2nd conv layer which has more in channels + out channels on x tensor)

        x = x.view(-1, int(x.nelement() / x.shape[0])) #view(*shape(returns a new tensor with the same data but a different shape)) were the shape is -1, the nelement of x, and the first dim of x
        x = F.relu(self.fc1(x)) #applies rectified linear unit on the fc1 layer applied on x
        x = F.relu(self.fc2(x)) #applies relu on the fc2 layer applied on the x tensor

        x = self.fc3(x) #applies the linear projection of 84, to 10
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet5().to(device=device)