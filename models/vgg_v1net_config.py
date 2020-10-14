"""Config file for VGG."""

class ConfigDict(object):
  pass

def vgg_v1net_config(vgg_depth=16,
               v1_timesteps=6,
               v1_kernel_size=5,
               ):
  """Return configuration to build VGG."""
  config = ConfigDict()
  config.image_size = 224
  config.vgg_depth = vgg_depth
  config.ckpt_dir = "pretrained_nets/vgg_%s" % config.vgg_depth
  config.num_classes = 1001
  config.add_v1net = True
  config.v1_timesteps = v1_timesteps
  config.v1_kernel_size = v1_kernel_size
  return config