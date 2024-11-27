import torch
import torch.nn as nn
import torch.nn.functional as F


def cin(x, scale, shift):
    mean = x.mean([2, 3], keepdim=True)
    var = x.var([2, 3], unbiased=False, keepdim=True) + 1e-5        # 多加了1e-5
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



class v39_FCM(nn.Module):
    def __init__(self, channel_size, begin_param, end_param):
        super(v39_FCM, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=1)
        self.fc1 = nn.Linear(channel_size // 2, channel_size // 2)
        self.fc2 = nn.Linear(channel_size // 2, channel_size // 2)
        self.conv2 = nn.Conv2d(channel_size // 2, channel_size, kernel_size=1)

        self.begin_param = begin_param
        self.end_param = end_param

    def forward(self, x):
        feature = x[0]
        guide = x[1]
        diff_weight = x[2]

        x = self.conv1(feature)

        scale = guide[:, self.begin_param : self.begin_param+(self.end_param-self.begin_param)//2]
        scale = self.fc2(F.relu(self.fc1(scale)))
        scale = scale.view(-1, x.size(1), 1, 1)

        shift = guide[:, self.begin_param+(self.end_param-self.begin_param)//2 : self.end_param]
        shift = self.fc2(F.relu(self.fc1(shift)))
        shift = shift.view(-1, x.size(1), 1, 1)

        x = x * diff_weight + x
        x = cin(x, scale, shift)
        delta_f = self.conv2(x)
        return feature + delta_f  # Residual connection


class v40_FCM(nn.Module):
    def __init__(self, channel_size, begin_param, end_param):
        super(v40_FCM, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=1)
        self.fc1 = nn.Linear(channel_size // 2, channel_size // 2)
        self.fc2 = nn.Linear(channel_size // 2, channel_size // 2)
        self.conv2 = nn.Conv2d(channel_size // 2, channel_size, kernel_size=1)

        self.begin_param = begin_param
        self.end_param = end_param

    def forward(self, x):
        feature = x[0]
        guide = x[1]
        diff_weight = x[2]

        x = self.conv1(feature)

        scale = guide[:, self.begin_param : self.begin_param+(self.end_param-self.begin_param)//2]
        scale = self.fc2(F.relu(self.fc1(scale)))
        scale = scale.view(-1, x.size(1), 1, 1)

        shift = guide[:, self.begin_param+(self.end_param-self.begin_param)//2 : self.end_param]
        shift = self.fc2(F.relu(self.fc1(shift)))
        shift = shift.view(-1, x.size(1), 1, 1)

        x = cin(x, scale, shift)
        x = x * diff_weight + x
        delta_f = self.conv2(x)
        return feature + delta_f  # Residual connection


class v42_FCM(nn.Module):
    def __init__(self, channel_size, begin_param, end_param):
        super(v42_FCM, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=1)
        self.fc1 = nn.Linear(channel_size // 2, channel_size // 2)
        self.fc2 = nn.Linear(channel_size // 2, channel_size // 2)
        self.conv2 = nn.Conv2d(channel_size // 2, channel_size, kernel_size=1)

        self.begin_param = begin_param
        self.end_param = end_param

    def forward(self, x):
        feature = x[0]
        guide = x[1]
        if feature.size(1) // 2 == x[2][0].size(1):
            diff_weight = x[2][0]
        else:
            diff_weight = x[2][1]

        x = self.conv1(feature)

        scale = guide[:, self.begin_param : self.begin_param+(self.end_param-self.begin_param)//2]
        scale = self.fc2(F.relu(self.fc1(scale)))
        scale = scale.view(-1, x.size(1), 1, 1)

        shift = guide[:, self.begin_param+(self.end_param-self.begin_param)//2 : self.end_param]
        shift = self.fc2(F.relu(self.fc1(shift)))
        shift = shift.view(-1, x.size(1), 1, 1)

        x = cin(x, scale, shift)
        x = x * diff_weight + x
        delta_f = self.conv2(x)
        return feature + delta_f  # Residual connection


class b_v35_5_FCM(nn.Module):
    def __init__(self, channel_size):
        super(b_v35_5_FCM, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size // 2, kernel_size=1)
        # self.fc1 = nn.Linear(channel_size // 2, channel_size // 2)
        # self.fc2 = nn.Linear(channel_size // 2, channel_size // 2)
        self.conv2 = nn.Conv2d(channel_size // 2, channel_size, kernel_size=1)

        # self.begin_param = begin_param
        # self.end_param = end_param

    def forward(self, x):
        feature = self.conv1(x)
        delta_f = self.conv2(feature)
        return x + delta_f  # Residual connection


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class v51_Conv_FCM(nn.Module):
    def __init__(self, c1, c2, begin_param_low, end_param_low, begin_param_high, end_param_high, k=1, s=1, p=None, g=1, act=True):
        super(v51_Conv_FCM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.fcm_low = FCM(c2, begin_param_low, end_param_low)
        self.fcm_high = FCM(c2, begin_param_high, end_param_high)
        self.conv_2 = nn.Conv2d(c2, c2, kernel_size=1)

    def forward(self, x):
        en_low = x[0][0]
        en_high = x[0][1]
        param = x[1]
        en_low = self.act(self.bn(self.conv(en_low)))
        en_low = self.fcm_low([en_low, param])
        en_high = self.act(self.bn(self.conv(en_high)))
        en_high = self.fcm_high([en_high, param])
        fused_feature = en_low + en_high
        return fused_feature


class v52_Conv_FCM(nn.Module):
    def __init__(self, c1, c2, begin_param_low, end_param_low, begin_param_high, end_param_high, k=1, s=1, p=None, g=1, act=True):
        super(v52_Conv_FCM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.fcm_low = FCM(c2, begin_param_low, end_param_low)
        self.fcm_high = FCM(c2, begin_param_high, end_param_high)
        self.conv_2 = nn.Conv2d(c2 * 2, c2, kernel_size=1)

    def forward(self, x):
        en_low = x[0][0]
        en_high = x[0][1]
        param = x[1]
        en_low = self.act(self.bn(self.conv(en_low)))
        en_low = self.fcm_low([en_low, param])
        en_high = self.act(self.bn(self.conv(en_high)))
        en_high = self.fcm_high([en_high, param])
        fused_feature = torch.cat([en_low, en_high], dim=1)
        fused_feature = self.conv_2(fused_feature)
        return fused_feature
