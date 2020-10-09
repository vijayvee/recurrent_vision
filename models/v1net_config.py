"""Config file for shallow V1Net CNN."""

class ConfigDict(object):
  pass

def v1net_config():
  """Return configuration to build shallow V1Net cnn."""
  config = ConfigDict()
  config.image_size = 256
  config.timesteps = 7
  config.num_classes = 2
  config.v1net_filters = 32
  config.v1net_kernel_size = 5
  return config