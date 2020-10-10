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
    net, _ = model_fn(inputs=net,
                   num_classes=model_config.num_classes,
                   is_training=is_training,
                   )
    return net
  
  def restore_checkpoint(self, checkpoint_path):
    """Function to restore weights from checkpoint."""
    pass
