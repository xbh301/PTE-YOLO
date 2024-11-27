import torch
import torch.nn as nn
import torch.nn.functional as F


class input(nn.Module):
    def __init__(self):
        super(input, self).__init__()

    def forward(self, x):
        return x


class b_v22_1_input(nn.Module):
    def __init__(self):
        super(b_v22_1_input, self).__init__()

    def forward(self, x):
        return x[0]



