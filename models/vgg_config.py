"""Config file for VGG."""
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
mean_rgb_vgg = np.array([_R_MEAN, _G_MEAN, _B_MEAN])

class ConfigDict(object):
  pass

def vgg_config(vgg_depth=16):
  """Return configuration to build VGG."""
  config = ConfigDict()
  config.image_size = (224, 224)
  config.model_name = "vgg_%s" % vgg_depth
  config.vgg_depth = vgg_depth
  config.mean_rgb = mean_rgb_vgg
  config.ckpt_dir = "pretrained_nets/vgg_%s" % config.vgg_depth
  config.num_classes = 1000
  config.add_v1net = False
  return config
