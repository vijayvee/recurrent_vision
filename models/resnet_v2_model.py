"""Builder for ResNet."""
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import tf_slim as slim  # pylint: disable=import-error
from recurrent_vision.models.model_builder import ModelBuilder
from recurrent_vision.models.pretrained_nets import resnet_v2


class ResNetV2(ModelBuilder):
  """Class for building ResNet-V2 CNNs."""
  def __init__(self, model_config):
    self.model_config = model_config
    self.image_size = model_config.image_size
    self.resnet_depth = model_config.resnet_depth    
    
  def preprocess(self, images):
    """Model-specific preprocessing of input images."""
    images = images - self.model_config.mean_rgb
    return images

  def to_1000_classes(self, logits):
    """Convert 1001 class logits to 1000 classes."""
    _, n = logits.shape.as_list()
    if n == 1001:
      # Discard logit 0 corresponding to background class
      logits = logits[:, 1:]
    return logits

  def build_model(self, images, is_training=True, preprocess=True):
    """Build model with input images."""
    net = tf.identity(images)
    if preprocess:
      net = self.preprocess(net)
    model_config = self.model_config
    if self.resnet_depth == 50:
      model_fn = resnet_v2.resnet_v2_50
    elif self.resnet_depth == 101:
      model_fn = resnet_v2.resnet_v2_101
    elif self.resnet_depth == 152:
      model_fn = resnet_v2.resnet_v2_152
    elif self.resnet_depth == 200:
      model_fn = resnet_v2.resnet_v2_200
    all_vars = [var for var in tf.global_variables()]
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, endpoints = model_fn(inputs=net,
                                num_classes=model_config.num_classes,
                                is_training=is_training,
                                add_v1net=model_config.add_v1net,
                                add_v1net_early=model_config.add_v1net_early,
                                )
    model_vars = [var for var in tf.global_variables()]
    self.model_vars = set(model_vars).difference(set(all_vars))
    net = self.to_1000_classes(net)
    return net, endpoints

  def restore_checkpoint(self, sess, latest_checkpoint):
    """Function to restore weights from checkpoint."""
    if not latest_checkpoint:
      latest_checkpoint = self.model_config.checkpoint_path
    # Get model vars
    model_vars = [var.name[:-2] for var in list(self.model_vars)]
    # Get checkpoint vars
    ckpt_vars = list(tf.train.list_variables(latest_checkpoint))
    ckpt_vars = [var for var, var_shape in ckpt_vars]
    # Restore vars = checkpoint_vars.intersection(model_vars)
    restore_vars = list(set(ckpt_vars).intersection(set(model_vars)))
    print("Restoring %s variables" % len(restore_vars))
    restore_vars = [var for var in self.model_vars 
                    if var.name[:-2] in restore_vars]
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, latest_checkpoint)