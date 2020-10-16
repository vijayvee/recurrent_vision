"""Config file for VGG."""

class ConfigDict(object):
  pass

def vgg_config(vgg_depth=16):
  """Return configuration to build VGG."""
  config = ConfigDict()
  config.image_size = (224, 224)
  config.model_name = "vgg_%s" % vgg_depth
  config.vgg_depth = vgg_depth
  config.ckpt_dir = "pretrained_nets/vgg_%s" % config.vgg_depth
  config.num_classes = 1000
  config.add_v1net = False
  return config
