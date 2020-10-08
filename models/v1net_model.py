"""Builder for V1Net CNNs."""
import tensorflow as tf
from recurrent_vision.models.model_builder import ModelBuilder
from recurrent_vision.utils.model_utils import build_avgpool, build_dense, build_v1net


class V1NetCNN(ModelBuilder):
  """Class for building V1Net CNN."""
  def __init__(self, model_config):
    self.model_config = model_config
    self.image_size = model_config.image_size
    
  def preprocess(self, images):
    """Model-specific preprocessing of input images."""
    images = tf.image.resize_bilinear(images, 
                                      self.image_size)
    return images

  def build_model(self, images, is_training=True):
    """Build model with input images."""
    with tf.variable_scope("V1NetCNN", reuse=tf.AUTO_REUSE):
      model_config = self.model_config
      net = self.preprocess(images)
      net = self.convolution_stem(net)
      # TODO(vveerabadran): create test for V1Net
      net = build_v1net(inputs=net, 
                          timesteps=model_config.timesteps,
                          filters=model_config.v1net_filters,
                          kernel_size=model_config.v1net_kernel_size,
                          is_training=is_training,
                          )
      net = build_avgpool(net)
      net = build_dense(net, units=2)
    return net
  
  def restore_checkpoint(self, checkpoint_path):
    """Function to restore weights from checkpoint."""
    pass