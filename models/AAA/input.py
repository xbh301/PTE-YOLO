import torch
import torch.nn as nn
import torch.nn.functional as F


class input(nn.Module):
    def __init__(self):
        super(input, self).__init__()

    def forward(self, x):
        return x
