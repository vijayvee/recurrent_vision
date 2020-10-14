"""Builder for ResNet."""
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
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
    images = tf.image.resize(images, 
                             [self.image_size,
                             self.image_size],
                             )
    return images

  def build_model(self, images, is_training=True):
    """Build model with input images."""
    net = self.preprocess(images)
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
    net, _ = model_fn(inputs=net,
                   num_classes=model_config.num_classes,
                   is_training=is_training,
                   add_v1net=model_config.add_v1net,
                  #  v1_timesteps=model_config.v1_timesteps,
                  #  v1_kernel_size=model_config.v1_kernel_size,
                   )
    model_vars = [var for var in tf.global_variables()]
    self.model_vars = set(all_vars).difference(set(model_vars))
    return net
  
  def restore_checkpoint(self, sess, checkpoint_path):
    """Function to restore weights from checkpoint."""
    if not checkpoint_path:
      checkpoint_path = self.model_config.checkpoint_path
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver = tf.train.Saver(self.model_vars)
    saver.restore(sess, latest_checkpoint)
