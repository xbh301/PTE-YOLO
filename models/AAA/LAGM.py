import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 98)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        x = self.fc(x)
        return x


class LAGM_v2(nn.Module):         # Light Aware Guidance Module         卷积核s=2 的下采样换成pooling层
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),          #####   stride应该写1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 98)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        x = self.fc(x)
        return x


class v5_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 101)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        x = self.fc(x)
        return x


class v8_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 10)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        x = self.fc(x)
        return x


def CropAndProcess(x):
    batch_size, channels, height, width = x.size()
    crop_height, crop_width = height // 2, width // 2

    crops = []
    for i in range(2):
        for j in range(2):
            crop = x[:, :, i * crop_height : (i + 1) * crop_height, j * crop_width : (j + 1) * crop_width]
            crops.append(crop)

    # Convert list to tensor with batch dimension
    # crops = torch.cat(crops, dim=0)
    return crops


class v12_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64*4, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 98)                     # 输出维度需要调整
        )

    def forward(self, x):
        x_patch = CropAndProcess(x)
        features = []
        for x in x_patch:
            # x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            x = self.cbl1(x)
            x = self.cbl2(x)
            x = self.cbl3(x)
            x = self.cbl4(x)
            features.append(x)
        x = torch.cat(features, 1)
        x = self.cbl5(x)
        x = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        x = self.fc(x)
        return x


def RandomCropSingleImage(x):
    batch_size, channels, height, width = x.size()
    crop_height, crop_width = height // 2, width // 2

    # i = random.randint(0, crop_height)
    # j = random.randint(0, crop_width)
    # crop = x[:, :, i:i + crop_height, j:j + crop_width]

    crop = x[:, :, height//4 : height//4+crop_height, width//4 : width//4+crop_width]

    return crop


class v13_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 98)                     # 输出维度需要调整
        )

    def forward(self, x):
        x_patch = RandomCropSingleImage(x)

        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(x_patch)))))
        x = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        x = self.fc(x)
        return x


class v14_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(x)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v14_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 98)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class v19_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 97)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class v20_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 97)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        x = self.fc(x)
        return x


class v25_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self,
                 num_blocks=[1, 1],
                 heads=[1, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.transformer1 = nn.Sequential(*[
            TransformerBlock(dim=16, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.transformer2 = nn.Sequential(*[
            TransformerBlock(dim=64, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.cbl5(self.transformer2(self.cbl4(self.cbl3(self.transformer1(self.cbl2(self.cbl1(x)))))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v26_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self,
                 num_blocks=[1, 1],
                 heads=[1, 4],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        # self.cbl2 = nn.Sequential(
        #     nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(inplace=True)
        # )
        self.transformer1 = nn.Sequential(*[
            TransformerBlock(dim=16, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        # self.cbl4 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(inplace=True)
        # )
        self.transformer2 = nn.Sequential(*[
            TransformerBlock(dim=64, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

        self.prompt1 = PromptGenBlock(prompt_dim=16, prompt_len=5, prompt_size=68, lin_dim=16)
        self.prompt2 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=34, lin_dim=64)

        self.pim_trans1 = TransformerBlock(dim=16 * 2, num_heads=heads[0],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)
        self.pim_trans2 = TransformerBlock(dim=64 * 2, num_heads=heads[1],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type)

        self.reduce_chan_level1 = nn.Conv2d(16 * 2, 16, kernel_size=1, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=bias)

        self.down1 = Downsample(8)
        self.down2 = Downsample(32)

    def forward(self, x):
        x = self.transformer1(self.down1(self.cbl1(x)))
        prompt1 = self.prompt1(x)
        x = torch.cat([x, prompt1], 1)
        x = self.reduce_chan_level1(self.pim_trans1(x))

        x = self.transformer2(self.down2(self.cbl3(x)))
        prompt2 = self.prompt2(x)
        x = torch.cat([x, prompt2], 1)
        x = self.reduce_chan_level2(self.pim_trans2(x))

        x = self.cbl5(x)
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v22_4_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),          #####   stride应该写1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(x)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v22_5_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),          #####   stride应该写1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(x)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v22_5_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 97)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x

class v22_6_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 184)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class v22_4_4_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 100)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class b_v22_1_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 96)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class b_v22_2_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 33)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class b_v22_3_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 65)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class b_v22_4_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)  # 输出维度需要调整
        )

    def forward(self, x):
        embedding = x[0]
        x = self.fc(embedding)
        return x


class v27_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(x)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v27_1_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(x)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class PreActResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActResBlock, self).__init__()

        # 预激活结构：先BatchNorm和ReLU，再卷积
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(x)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += shortcut
        return out


class v29_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(inplace=True)
        )

        self.resblock1 = PreActResBlock(16, 16, stride=1)
        self.resblock2 = PreActResBlock(16, 32, stride=1)
        self.resblock3 = PreActResBlock(32, 64, stride=2)
        self.resblock4 = PreActResBlock(64, 128, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.resblock4(self.resblock3(self.resblock2(self.resblock1(x))))
        x = self.avg_pool(x)
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class RCAB(nn.Module):
    #残差通道注意力块
    def __init__(self, in_channel, act=nn.LeakyReLU(negative_slope=0.2),):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(*[
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
            act,
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True),
            CALayer(in_channel, True)
        ])

    def forward(self, x):
        # x+x*y，输出为 输入+乘以通道注意力后的输入
        out = self.body(x)
        out += x
        return out


class RG(nn.Module):
    #残差通道注意力组，RCAB的组合
    def __init__(self, num_RCAB, inchannel):
        super(RG, self).__init__()
        body = []
        for i in range(num_RCAB):
            body.append(RCAB(in_channel=inchannel))
        body.append(nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=(3 - 1) // 2, stride=1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        out += x
        #输入加上 n个RCAB级联后经过3*3卷积
        return out


class CALayer(nn.Module):
    # Channel Attention (CA) Layer
    # 通道注意力层，输出为 输入*通道注意力
    def __init__(self, channel, bias):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point，一个通道化为一个点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化，指定输出尺寸
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // 16, 1, padding=0, bias=bias),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel // 16, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, stride=1, expand_ratio=6, activation=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride in [1, 2]
        hidden_dim = int(in_channels * expand_ratio)
        self.is_residual = self.stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # pw Point-wise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            activation,
            # dw  Depth-wise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            activation,
            # pw-linear, Point-wise linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),

        )

    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            res = self.conv(x)
            x = x + res
        else:
            x = self.conv(x)
        return x


class v30_LAGM(nn.Module):
    #sRGB分支的引导模块
    def __init__(self):
        super(v30_LAGM, self).__init__()
        #先上一个3*3卷积，输出64个通道
        self.head = nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=3, padding=(3 - 1) // 2, stride=2, bias=True),
        ])

        self.RG1 = RCAB(64)
        self.MV2_1 = InvertedResidualBlock(in_channels=64, out_channels=128, stride=2, expand_ratio=6)
        self.RG2 = RCAB(128)
        self.MV2_2 = InvertedResidualBlock(in_channels=128, out_channels=128, stride=1, expand_ratio=6)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )


    def forward(self, x):
        head = self.head(x)  # (H, W, 32)
        x = self.RG1(head) + head
        x = self.MV2_1(x)
        x = self.RG2(x) + x
        x = self.MV2_2(x)
        x = self.avg_pool(x)
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v31_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

        self.is_contrast_training = False

    def forward(self, x):
        if not self.is_contrast_training:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(x)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v27_2_LAGM(nn.Module):
    #sRGB分支的引导模块
    def __init__(self):
        super(v27_2_LAGM, self).__init__()
        #先上一个3*3卷积，输出64个通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))

        self.RG1 = RCAB(64)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))

        self.RG2 = RCAB(128)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        x = self.conv4(self.RG2(self.conv3(self.RG1(self.conv2(self.conv1(x))))))
        x = self.avg_pool(x)
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        out = self.fc(fea)
        return fea, out


class v27_3_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)

        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(img_patches)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


class v27_3_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 97)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class v27_4_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)

        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(img_patches)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


def random_resized_crop(x, size, scale=(0.06, 1.0), ratio=(1, 1), num_crops=5):
    B, C, orig_h, orig_w = x.size()
    support_set = []

    for i in range(B):
        img_patches = []
        for _ in range(num_crops):
            area = orig_h * orig_w
            target_area = area * torch.empty(1).uniform_(*scale).item()
            aspect_ratio = torch.empty(1).uniform_(*ratio).item()

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if h <= orig_h and w <= orig_w:
                i_start = torch.randint(0, orig_h - h + 1, size=(1,)).item()
                j_start = torch.randint(0, orig_w - w + 1, size=(1,)).item()

                cropped = x[i, :, i_start:i_start+h, j_start:j_start+w]
                a = cropped.unsqueeze(0)
                resized = F.interpolate(cropped.unsqueeze(0), size=size, mode='bilinear', align_corners=False)
                img_patches.append(resized)
            else:
                resized = F.interpolate(x[i].unsqueeze(0), size=size, mode='bilinear', align_corners=False)
                img_patches.append(resized)
        img_patches = torch.cat(img_patches, dim=1)
        support_set.append(img_patches)

    support_set = torch.cat(support_set, dim=0)
    return support_set


class v27_9_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )


    def forward(self, x):
        img_patches = random_resized_crop(x, size=x.shape[3] // 4)

        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(img_patches)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


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
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)

        x = self.downsample(self.cbl1(img_patches))
        x = self.downsample(self.cbl2(x))
        x = self.cbl4(self.cbl3(x))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


class v36_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)

        x = self.cbl5(self.cbl4(self.cbl3(self.cbl2(self.cbl1(img_patches)))))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


class v37_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.cbl3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.downsample = nn.AvgPool2d(2, 2)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)

        x = self.downsample(self.cbl1(img_patches))
        x = self.downsample(self.cbl2(x))
        x = self.cbl3(x)
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


class b_v35_1_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 96)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class b_v35_2_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class b_v35_3_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 33)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class b_v35_4_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 65)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class b_v35_6_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        img_patches = x[1]

        x = self.downsample(self.cbl1(img_patches))
        x = self.downsample(self.cbl2(x))
        x = self.cbl4(self.cbl3(x))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


class v44_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 110)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class v46_LAGM(nn.Module):         # Light Aware Guidance Module
    def __init__(self):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True))
        self.cbl2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(128, 128)                     # 输出维度需要调整
        )

    def forward(self, x):
        img_patches = x[1]
        unbound = torch.unbind(img_patches, dim=1)
        img_patches = torch.cat(unbound, dim=1)

        x = self.downsample(self.cbl1(img_patches))
        x = self.downsample(self.cbl2(x))
        x = self.cbl4(self.cbl3(x))
        fea = x.view(x.size(0), -1)  # 展平操作，改变形状为[batch_size, 128]
        # out = self.fc(fea)
        return fea


class v47_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 98)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class v51_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 130)  # 输出维度需要调整
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class v54_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 120)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x


class b_v35_7_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 994)  # 输出维度需要调整
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class b_v35_8_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 897)  # 输出维度需要调整
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class v55_fc_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 120)  # 输出维度需要调整
        )

    def forward(self, x):
        # embedding = x[0]
        x = self.fc(x)
        return x

