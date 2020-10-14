"""Config file for ResNet_v2 + V1Net CNN."""

class ConfigDict(object):
  pass

def resnet_v1net_config():
  """Return configuration to build ResNet + V1Net cnn."""
  config = ConfigDict()
  config.image_size = 299
  config.timesteps = 7
  config.v1net_filters = 32
  config.v1net_kernel_size = 5
  config.resnet_depth = 50
  config.num_classes = 1000
  return config