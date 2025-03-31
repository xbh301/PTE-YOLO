import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torchvision.utils import save_image
import os



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


class v60_ReciprocalCurveFilter(Filter):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.begin_filter_parameter = cfg.reciprocal_curve_begin_param
    self.num_filter_parameters = 3

  def filter_param_regressor(self, features):
    log_alpha_left = np.log(self.cfg.alpha_range[0])
    log_alpha_right = np.log(self.cfg.alpha_range[1])
    log_alpha = tanh_range(log_alpha_left, log_alpha_right)(features)
    alpha = torch.exp(log_alpha)  
    return alpha

  def process(self, img, param):
    # param = param+0.2
    alpha = param[:, :, None, None]
    enhanced_img = (alpha + 1) * img / (img + alpha)
    return enhanced_img


class v60_DIP(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.reciprocal_curve = v60_ReciprocalCurveFilter(cfg)

  def forward(self, img, param):
    img = self.reciprocal_curve.apply(img, param)
    return img


class v60_ImageLevelEnhancement(nn.Module):
  def __init__(self):
    super().__init__()
    self.num_filter_parameters = 3
    self.reciprocal_curve_begin_param = 0
    self.alpha_range = (0.001, 100.0)
    self.GAP = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(16, 16),
      nn.LeakyReLU(inplace=True),
      nn.Linear(16, 3) 
    )

    self.dip = v60_DIP(self)

  def forward(self, x):
    param = x[0][0]              
    low = x[1][0]
    param = self.fc(self.GAP(param).view(low.size(0), -1))
    en = self.dip(low, param).type_as(low)
    return en
