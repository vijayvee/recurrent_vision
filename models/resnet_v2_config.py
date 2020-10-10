"""Config file for ResNet_v2."""

class ConfigDict(object):
  pass

def resnet_v2_config():
  """Return configuration to build ResNet."""
  config = ConfigDict()
  config.image_size = 299
  config.resnet_depth = 50
  config.ckpt_dir = "pretrained_nets/resnet_v2_%s" % config.resnet_depth
  config.num_classes = 1001
  return config