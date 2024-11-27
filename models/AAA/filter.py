import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torchvision.utils import save_image
import os

def save_intermediate_image(x, filename):
  # 创建保存图片的文件夹
  save_dir = "E:/results/images/1111_en_image"
  # os.makedirs(save_dir, exist_ok=True)

  # 将Tensor数据范围从[-1, 1]或其他范围映射到[0, 1]
  x = (x - x.min()) / (x.max() - x.min())

  # 使用 torchvision.utils.save_image 保存图像
  save_image(x, os.path.join(save_dir, filename))

def rgb2lum(image):
  image = 0.27 * image[:,
                 0, :, :] + 0.67 * image[:,
                                   1, :, :] + 0.06 * image[:,
                                                     2, :, :]
  return image[:, None, :, :]


def tanh01(x):
  return torch.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):
  def get_activation(left, right, initial):

    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left

    return activation

  return get_activation(l, r, initial)


def lerp(a, b, l):
  return (1 - l) * a + l * b


class Filter:
  def __init__(self, cfg):
    self.cfg = cfg
    # Specified in child classes
    self.begin_filter_parameter = None
    self.num_filter_parameters = None
    self.filter_parameters = None

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def get_begin_filter_parameter(self):
    return self.begin_filter_parameter

  def extract_parameters(self, features):
    return features[:,
           self.get_begin_filter_parameter():(
                   self.get_begin_filter_parameter() +
                   self.get_num_filter_parameters())]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    pass

  # Should be implemented in child classes
  def process(self, img, param):
    pass

  # Apply the whole filter
  def apply(self, img=None, img_features=None):
    assert img is not None
    assert img_features is not None
    filter_features = self.extract_parameters(img_features)
    filter_parameters = self.filter_param_regressor(filter_features)
    low_res_output = self.process(img, filter_parameters)

    return low_res_output


class ImprovedWhiteBalanceFilter(Filter):

  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.channels = 3
    self.begin_filter_parameter = cfg.wb_begin_param
    self.num_filter_parameters = self.channels

  def filter_param_regressor(self, features):
    log_wb_range = 0.5

    mask = torch.tensor(((0, 1, 1)),
                        dtype=torch.float32,
                        device=features.device).reshape(1, 3)
    assert mask.shape == (1, 3)

    features = features * mask
    features = tanh_range(-log_wb_range, log_wb_range)(features)
    color_scaling = torch.exp(features.clone()).clone()
    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    color_scaling *= 1.0 / (1e-5 + 0.27 * color_scaling[:, 0] +
                            0.67 * color_scaling[:, 1] +
                            0.06 * color_scaling[:, 2])[:, None]
    return color_scaling

  def process(self, img, param):
    return img * param[:, :, None, None]


class GammaFilter(Filter):
  # gamma_param is in [1/gamma_range, gamma_range]
  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.begin_filter_parameter = cfg.gamma_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):                             
    log_gamma_range = np.log(self.cfg.gamma_range)
    return torch.exp(
      tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param):
    param_1 = param.repeat([1, 3])
    return torch.pow(
      torch.maximum(img, torch.tensor([0.001], device=img.device)),
      param_1[:, :, None, None])


class ToneFilter(Filter):

  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.curve_steps = cfg.curve_steps
    self.begin_filter_parameter = cfg.tone_begin_param

    self.num_filter_parameters = cfg.curve_steps

  def filter_param_regressor(self, features):
    tone_curve = torch.reshape(features,
                               shape=(-1, 1, self.cfg.curve_steps))[:, :,
                 None,
                 None]
    tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
    return tone_curve

  def process(self, img, param):
    # img = tf.minimum(img, 1.0)
    tone_curve = param
    tone_curve_sum = torch.sum(tone_curve, dim=4) + 1e-30
    total_image = img * 0
    for i in range(self.cfg.curve_steps):
      total_image += torch.clamp(
        img - 1.0 * i / self.cfg.curve_steps, 0,
        1.0 / self.cfg.curve_steps) * param[:, :, :, :, i]
    total_image *= self.cfg.curve_steps / tone_curve_sum
    total_image = total_image.type_as(img)
    img = total_image
    return img


class ContrastFilter(Filter):
  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.begin_filter_parameter = cfg.contrast_begin_param

    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return torch.tanh(features)

  def process(self, img, param):
    luminance = torch.minimum(
      torch.maximum(rgb2lum(img), torch.tensor([0.0],
                                               device=img.device)),
      torch.tensor([1.0], device=img.device))
    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    return lerp(img, contrast_image, param[:, :, None, None])


class UsmFilter(Filter):
  def __init__(self, cfg):
    Filter.__init__(self, cfg)
    self.begin_filter_parameter = cfg.usm_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(*self.cfg.usm_range)(features)

  def process(self, img, param):
    def make_gaussian_2d_kernel(sigma, dtype=torch.float32, device=None):
      radius = 12
      x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
      k = torch.exp(-0.5 * torch.square(x / sigma))
      k = k / torch.sum(k)
      return torch.unsqueeze(k, 1) * k

    kernel_i = make_gaussian_2d_kernel(5, dtype=img.dtype, device=img.device)
    kernel_i = kernel_i.repeat([3, 1, 1, 1])

    pad_w = (25 - 1) // 2
    padded = torch.nn.functional.pad(img, (pad_w, pad_w, pad_w, pad_w),
                                     mode='reflect')

    output = torch.nn.functional.conv2d(padded,
                                        weight=kernel_i,
                                        stride=(1, 1), groups=3)

    img_out = (img - output) * param[:, :, None, None] + img

    return img_out


class ReciprocalCurveFilter(Filter):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.begin_filter_parameter = cfg.reciprocal_curve_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    log_alpha_left = np.log(self.cfg.alpha_range[0])
    log_alpha_right = np.log(self.cfg.alpha_range[1])
    log_alpha = tanh_range(log_alpha_left, log_alpha_right)(features)
    alpha = torch.exp(log_alpha)  # 将参数进行指数变换
    return alpha

  def process(self, img, param):
    # param = param+0.2
    alpha = param.repeat([1, 3])
    alpha = alpha[:, :, None, None]
    enhanced_img = (alpha + 1) * img / (img + alpha)
    return enhanced_img


class DIP(nn.Module):         # 参照iayolo，只使用gamma变换和contrast

  def __init__(self, cfg):
    super().__init__()

    # self.wb = ImprovedWhiteBalanceFilter(cfg)
    self.gamma = GammaFilter(cfg)
    # self.tone = ToneFilter(cfg)
    self.contrast = ContrastFilter(cfg)
    # self.sharpen = UsmFilter(cfg)

  def forward(self, img, param):
    # img = self.wb.apply(img, param)
    img = self.gamma.apply(img, param)
    # img = self.tone.apply(img, param)
    img = self.contrast.apply(img, param)
    # img = self.sharpen.apply(img, param)

    return img


class ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 2
    self.gamma_begin_param = 0
    self.contrast_begin_param = 1

    self.gamma_range = 2.5

    self.dip = DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1]
    feature = self.dip(feature, param).type_as(feature)
    return feature




class DIP_v2(nn.Module):         # 参照iayolo，只使用gamma变换和sharpen
  def __init__(self, cfg):
    super().__init__()
    self.gamma = GammaFilter(cfg)
    self.sharpen = UsmFilter(cfg)

  def forward(self, img, param):
    img = self.gamma.apply(img, param)
    img = self.sharpen.apply(img, param)

    return img


class ImageLevelEnhancement_v2(nn.Module):        # 对应实验v3版本
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 2
    self.gamma_begin_param = 0
    self.usm_begin_param = 1

    self.gamma_range = 2.5
    self.usm_range = (0.0, 2.5)

    self.dip = DIP_v2(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1]
    feature = self.dip(feature, param).type_as(feature)
    return feature


class v4_DIP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.contrast = ContrastFilter(cfg)
    self.sharpen = UsmFilter(cfg)

  def forward(self, img, param):
    img = self.contrast.apply(img, param)
    img = self.sharpen.apply(img, param)
    return img


class v4_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 2
    self.contrast_begin_param = 0
    self.usm_begin_param = 1

    self.usm_range = (0.0, 2.5)

    self.dip = v4_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1]
    feature = self.dip(feature, param).type_as(feature)
    return feature


class v5_DIP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tone = ToneFilter(cfg)
    self.sharpen = UsmFilter(cfg)

  def forward(self, img, param):
    img = self.tone.apply(img, param)
    img = self.sharpen.apply(img, param)

    return img


class v5_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 5
    self.tone_begin_param = 0
    self.usm_begin_param = 4

    self.usm_range = (0.0, 2.5)
    self.tone_curve_range = (0.5, 2)
    self.curve_steps = 4

    self.dip = v5_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1]
    feature = self.dip(feature, param).type_as(feature)
    return feature


class v14_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 2
    self.gamma_begin_param = 0
    self.usm_begin_param = 1

    self.gamma_range = 2.5
    self.usm_range = (0.0, 2.5)

    self.dip = DIP_v2(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1][0]
    feature = self.dip(feature, param).type_as(feature)
    return feature


class v19_DIP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.reciprocal_curve = ReciprocalCurveFilter(cfg)

  def forward(self, img, param):
    img = self.reciprocal_curve.apply(img, param)
    return img


class v19_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    low = x[1][0]
    en = self.dip(low, param).type_as(low)
    # save_intermediate_image(en, '22015_02799_v35.png')

    return en


class v20_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1]
    feature = self.dip(feature, param).type_as(feature)
    return feature


class v27_5_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1][0]
    feature = self.dip(feature, param).type_as(feature)
    out = torch.cat([x[1][0], feature], 1)
    return out


class SpatialAttention(nn.Module):
  def __init__(self, kernel_size=5):
    super().__init__()
    assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

    self.conv = nn.Conv2d(2,
                          1,
                          kernel_size,
                          padding=kernel_size // 2,
                          bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avgout = torch.mean(x, dim=1, keepdim=True)
    maxout, _ = torch.max(x, dim=1, keepdim=True)
    attention = torch.cat([avgout, maxout], dim=1)
    attention = self.conv(attention)
    return self.sigmoid(attention) * x


class Trans_guide(nn.Module):
  def __init__(self, ch=16):
    super().__init__()

    self.layer = nn.Sequential(
      nn.Conv2d(6, ch, 3, padding=1),
      nn.LeakyReLU(),
      # nn.ReLU(True),
      SpatialAttention(3),
      nn.Conv2d(ch, 3, 3, padding=1),
    )

  def forward(self, x):
    return self.layer(x)


class v27_6_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

    self.fusion = Trans_guide(32)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1][0]
    feature = self.dip(feature, param).type_as(feature)
    out = torch.cat([x[1][0], feature], 1)
    out = self.fusion(out)
    return out

class v27_9_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1]
    feature = self.dip(feature, param).type_as(feature)
    out = torch.cat([x[1], feature], 1)
    return out


class v33_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1][0]
    feature = self.dip(feature, param).type_as(feature)
    out = torch.cat([x[1][0], feature], 1)
    return out


class v34_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    feature = x[1][0]
    feature = self.dip(feature, param).type_as(feature)
    out = x[1][0] + feature
    return out


class v38_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.01, 10.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    low = x[1][0]
    en = self.dip(low, param).type_as(low)
    diff_img = (en - low).abs()
    out = torch.cat([en, diff_img], 1)
    return out


class b_v35_5_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.alpha = 0.1

  def forward(self, x):
    b, c, h, w = x.size()
    alpha = torch.full((b, 1), self.alpha, device=x.device, dtype=x.dtype)
    alpha = alpha.repeat([1, 3])
    alpha = alpha[:, :, None, None]
    enhanced_img = (alpha + 1) * x / (x + alpha)
    return enhanced_img


class v44_DIP(nn.Module):

  def __init__(self, cfg):
    super().__init__()

    self.wb = ImprovedWhiteBalanceFilter(cfg)
    self.gamma = GammaFilter(cfg)
    self.tone = ToneFilter(cfg)
    self.contrast = ContrastFilter(cfg)
    self.sharpen = UsmFilter(cfg)

  def forward(self, img, param):
    img = self.wb.apply(img, param)
    img = self.gamma.apply(img, param)
    img = self.tone.apply(img, param)
    img = self.contrast.apply(img, param)
    img = self.sharpen.apply(img, param)

    return img


class v44_iayolo(nn.Module):

  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 14
    # the begin index of param
    self.wb_begin_param = 0
    self.gamma_begin_param = 3
    self.tone_begin_param = 4
    self.contrast_begin_param = 12
    self.usm_begin_param = 13

    self.curve_steps = 4
    self.gamma_range = 2.5
    self.wb_range = 1.1
    self.tone_curve_range = (0.5, 2)
    self.usm_range = (0.0, 2.5)

    self.dip = v44_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    low = x[1][0]
    img = self.dip(low, param).type_as(param)
    return img


class v45_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 1
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.001, 100000.0)

    self.dip = v19_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    low = x[1][0]
    en = self.dip(low, param).type_as(low)
    # save_intermediate_image(en, 'conv1_output.jpg')
    return en


class v54_Filter:
  def __init__(self, cfg):
    self.cfg = cfg
    # Specified in child classes
    self.begin_filter_parameter = None
    self.num_filter_parameters = None
    self.filter_parameters = None

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def get_begin_filter_parameter(self):
    return self.begin_filter_parameter

  def extract_parameters(self, features):
    return features[:,
           self.get_begin_filter_parameter():(
                   self.get_begin_filter_parameter() +
                   self.get_num_filter_parameters())]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    pass

  # Should be implemented in child classes
  def process(self, img, param, param_local):
    pass

  # Apply the whole filter
  def apply(self, img=None, img_features=None, param_local=None):
    assert img is not None
    assert img_features is not None
    filter_features = self.extract_parameters(img_features)
    filter_parameters = self.filter_param_regressor(filter_features)
    low_res_output = self.process(img, filter_parameters, param_local)

    return low_res_output


class v54_ReciprocalCurveFilter(v54_Filter):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.begin_filter_parameter = cfg.reciprocal_curve_begin_param
    self.num_filter_parameters = 24
    self.tm_pts_num = 8

  def filter_param_regressor(self, features):
    log_alpha_left = np.log(self.cfg.alpha_range[0])
    log_alpha_right = np.log(self.cfg.alpha_range[1])
    log_alpha = tanh_range(log_alpha_left, log_alpha_right)(features)
    alpha = torch.exp(log_alpha)  # 将参数进行指数变换
    return alpha

  def process(self, img, param, param_local):
    n1, _, h1, w1 = img.shape
    n2, _, h2, w2 = param_local.shape
    assert n1 == n2 and h1 == h2 and w1 == w2, f'param_local has invalid shape < {param_local.shape} >!'

    ltm = tanh_range(0.1, 99.9)(param_local).reshape(n1, 3, self.tm_pts_num, h1, w1)  # 每个RGB通道分8段
    ltm = ltm / torch.sum(ltm, dim=2, keepdim=True)  # pieces
    ltm1_ = [ltm[:, :, :i].sum(2) for i in range(self.tm_pts_num + 1)]  # 计算累加和
    alpha = param.view(n1, 3, self.tm_pts_num, 1 ,1)
    alpha = alpha * ltm

    total_image = 0
    for i, alpha_local in enumerate(torch.split(alpha, 1, dim=2)):
      alpha_local = alpha_local.squeeze(2)
      local_image = torch.minimum(torch.clamp(img - ltm1_[i], min=0), ltm1_[i + 1] - ltm1_[i])
      local_image_en = (alpha_local + 1) * local_image / (local_image + alpha_local)
      total_image += local_image_en
    enhanced_img = (total_image - total_image.min()) / (total_image.max() - total_image.min())
    return enhanced_img


class v54_DIP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.reciprocal_curve = v54_ReciprocalCurveFilter(cfg)
    self.image_adaptive_local = nn.Sequential(
      nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 16, 3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 8 * 3, 3, stride=1, padding=1),
    )

  def forward(self, img, param):
    h, w = img.shape[-2:]
    param_local = self.image_adaptive_local(img)
    param_local = F.interpolate(param_local, size=(h, w), mode='bilinear')
    img = self.reciprocal_curve.apply(img, param, param_local)
    return img


class v54_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 24
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.001, 100.0)

    self.dip = v54_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    low = x[1][0]
    en = self.dip(low, param).type_as(low)
    return en


class v55_ReciprocalCurveFilter(v54_Filter):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.begin_filter_parameter = cfg.reciprocal_curve_begin_param
    self.num_filter_parameters = 24
    self.tm_pts_num = 8

  def filter_param_regressor(self, features):
    log_alpha_left = np.log(self.cfg.alpha_range[0])
    log_alpha_right = np.log(self.cfg.alpha_range[1])
    log_alpha = tanh_range(log_alpha_left, log_alpha_right)(features)
    alpha = torch.exp(log_alpha)  # 将参数进行指数变换
    return alpha

  def process(self, img, param, param_local):
    n1, _, h1, w1 = img.shape
    n2, _, h2, w2 = param_local.shape
    assert n1 == n2 and h1 == h2 and w1 == w2, f'param_local has invalid shape < {param_local.shape} >!'

    ltm = tanh_range(0.1, 99.9)(param_local).reshape(n1, 3, self.tm_pts_num, h1, w1)  # 每个RGB通道分8段
    ltm = ltm / torch.sum(ltm, dim=2, keepdim=True)  # pieces
    ltm1_ = [ltm[:, :, :i].sum(2) for i in range(self.tm_pts_num + 1)]  # 计算累加和
    alpha = param.view(n1, 3, self.tm_pts_num, 1 ,1)
    # alpha = alpha * ltm       # 不加权归一化
    alpha = alpha / torch.sum(ltm * alpha, dim=2, keepdim=True)   # 加权归一化

    total_image = torch.zeros_like(img, device=img.device)
    for i, alpha_local in enumerate(torch.split(alpha, 1, dim=2)):
      alpha_local = alpha_local.squeeze(2)
      # 当前分段的掩码
      mask = (img >= ltm1_[i]) & (img < ltm1_[i+1]) if i > 0 else (img < ltm1_[i+1])
      # 倒数曲线增强
      enhanced_segment = (alpha_local + 1) * img / (img + alpha_local)
      # 累加分段增强结果
      total_image += mask * enhanced_segment
    return total_image


class v55_DIP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.reciprocal_curve = v55_ReciprocalCurveFilter(cfg)
    self.image_adaptive_local = nn.Sequential(
      nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 16, 3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 8 * 3, 3, stride=1, padding=1),
    )

  def forward(self, img, param):
    h, w = img.shape[-2:]
    param_local = self.image_adaptive_local(img)
    param_local = F.interpolate(param_local, size=(h, w), mode='bilinear')
    img = self.reciprocal_curve.apply(img, param, param_local)
    return img


class v55_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 24
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.001, 100.0)

    self.dip = v55_DIP(self)

  def forward(self, x):
    param = x[0]                # 解包输入，适应yolov5结构
    low = x[1][0]
    en = self.dip(low, param).type_as(low)
    return en
