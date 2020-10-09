"""Builder for V1Net CNNs."""
import tensorflow as tf
from recurrent_vision.models.model_builder import ModelBuilder
from recurrent_vision.utils.model_utils import build_avgpool, build_dense, build_v1net


class V1NetCNN(ModelBuilder):
  """Class for building V1Net CNNs."""
  def __init__(self, model_config):
    self.model_config = model_config
    self.image_size = model_config.image_size
    
  def preprocess(self, images):
    """Model-specific preprocessing of input images."""
    images = tf.image.resize(images, 
                             (self.image_size,
                              self.image_size),
                             )
    return images

  def build_model(self, images, is_training=True):
    """Build model with input images."""
    with tf.variable_scope("V1NetCNN", reuse=tf.AUTO_REUSE):
      model_config = self.model_config
      num_classes = model_config.num_classes
      n, _, _, _ = images.shape.as_list()
      net = tf.identity(images)
      net = self.preprocess(net)
      net = self.convolution_stem(net)
      net = build_v1net(inputs=net, 
                        timesteps=model_config.timesteps,
                        filters=model_config.v1net_filters,
                        kernel_size=model_config.v1net_kernel_size,
                        is_training=is_training,
                        )
      net = build_avgpool(net)
      net = build_dense(net, units=2)
      net = tf.reshape(net, [n, num_classes])
    return net
  
  def restore_checkpoint(self, checkpoint_path):
    """Function to restore weights from checkpoint."""
    pass