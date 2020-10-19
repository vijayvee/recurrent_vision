"""Config file for VGG."""
import os
import numpy as np
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
mean_rgb_vgg = np.array([_R_MEAN, _G_MEAN, _B_MEAN])

class ConfigDict(object):
  pass


def vgg_16_hed_config(vgg_depth=16):
  """Return configuration to build VGG."""
  config = ConfigDict()
  config.image_size = (321, 481)
  config.model_name = "vgg_16_hed"
  config.vgg_depth = 16
  config.mean_rgb = mean_rgb_vgg
  base_ckpt_path = "models/pretrained_nets/checkpoints/"
  config.ckpt_dir = os.path.join(base_ckpt_path,
                                 "vgg_%s" % config.vgg_depth,
                                 "vgg_%s.ckpt" % config.vgg_depth
                                 )
  config.num_classes = 1
  config.add_v1net = False
  return config
