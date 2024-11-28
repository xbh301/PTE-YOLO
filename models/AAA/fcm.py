import torch
import torch.nn as nn
import torch.nn.functional as F


def cin(x, scale, shift):
    mean = x.mean([2, 3], keepdim=True)
    var = x.var([2, 3], unbiased=False, keepdim=True) + 1e-5     
    x_normalized = (x - mean) / torch.sqrt(var + 1e-5)
    return x_normalized * scale + shift


class FCM(nn.Module):
    def __init__(self, channel_size, begin_param, end_param):
        super(FCM, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=1)
        self.fc1 = nn.Linear(channel_size // 2, channel_size // 2)
        self.fc2 = nn.Linear(channel_size // 2, channel_size // 2)
        self.conv2 = nn.Conv2d(channel_size // 2, channel_size, kernel_size=1)

        self.begin_param = begin_param
        self.end_param = end_param

    def forward(self, x):
        feature = x[0]
        guide = x[1]

        x = self.conv1(feature)

        scale = guide[:, self.begin_param : self.begin_param+(self.end_param-self.begin_param)//2]
        scale = self.fc2(F.relu(self.fc1(scale)))
        scale = scale.view(-1, x.size(1), 1, 1)

        shift = guide[:, self.begin_param+(self.end_param-self.begin_param)//2 : self.end_param]
        shift = self.fc2(F.relu(self.fc1(shift)))
        shift = shift.view(-1, x.size(1), 1, 1)

        x = cin(x, scale, shift)
        delta_f = self.conv2(x)
        return feature + delta_f  # Residual connection
