"""Config file for ResNet_v2."""

class ConfigDict(object):
  pass

def resnet_v2_v1net_config():
  """Return configuration to build ResNet."""
  config = ConfigDict()
  config.image_size = (299, 299)
  config.resnet_depth = 50
  config.ckpt_dir = "pretrained_nets/resnet_v2_%s" % config.resnet_depth
  config.num_classes = 1001
  config.add_v1net = True
  config.timesteps = 7
  config.v1net_filters = 32
  config.v1net_kernel_size = 5
  return config