import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class v63_PG(nn.Module):         # Light Aware Guidance Module
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels*2),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(mid_channels*2, mid_channels*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels*4),
            nn.LeakyReLU(inplace=True)
        )
        self.downsample = nn.AvgPool2d(2, 2)

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)

        prompt_en = self.cbl2(self.cbl1(img_patches))
        prompt_layer1 = self.cbl3(self.downsample(prompt_en))
        prompt_layer2 = self.cbl4(self.downsample(prompt_layer1))

        return [prompt_en, prompt_layer1, prompt_layer2]
