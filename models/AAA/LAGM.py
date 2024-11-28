import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class v35_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.downsample = nn.AvgPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                    

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)
        x = self.downsample(self.cbl1(img_patches))
        x = self.downsample(self.cbl2(x))
        x = self.cbl4(self.cbl3(x))
        fea = x.view(x.size(0), -1) 
        return fea

class v27_3_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 97)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
