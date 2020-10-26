"""Config file for ResNet_v2."""
import numpy as np

mean_rgb = np.array([0.48501961, 
                     0.45796078, 
                     0.40760784])

class ConfigDict(object):
  pass

def resnet_v2_config(resnet_depth=50, 
                     add_v1net_early=False,
                     compact=False,
                     ):
  """Return configuration to build ResNet."""
  config = ConfigDict()
  config.image_size = (299, 299)
  config.resnet_depth = resnet_depth
  config.ckpt_dir = "pretrained_nets/resnet_v2_%s" % config.resnet_depth
  config.num_classes = 1001
  config.mean_rgb = mean_rgb
  config.add_v1net = False
  config.add_v1net_early = add_v1net_early
  config.compact = compact
  return config