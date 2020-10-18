"""Builder for VGG."""
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
from recurrent_vision.models.model_builder import ModelBuilder
from recurrent_vision.models.pretrained_nets import vgg


class VGG(ModelBuilder):
  """Class for building VGG CNNs."""
  def __init__(self, model_config):
    self.model_config = model_config
    self.image_size = model_config.image_size
    self.model_name = model_config.model_name
    
  def preprocess(self, images):
    """Model-specific preprocessing of input images."""
    images = tf.image.resize(images, 
                             self.image_size,
                             )
    return images

  def build_model(self, images, is_training=True):
    """Build model with input images."""
    net = self.preprocess(images)
    if is_training:
      net = self.augment_images(net)
    model_config = self.model_config
    if self.model_name.startswith("vgg_16_hed"):
      model_fn = vgg.vgg_16_hed
    elif self.model_name.startswith("vgg_19"):
      model_fn = vgg.vgg_19
    elif self.model_name.startswith("vgg_16"):
      model_fn = vgg.vgg_16
    all_vars = [var for var in tf.global_variables()]
    net, endpoints = model_fn(inputs=net,
                      num_classes=model_config.num_classes,
                      is_training=is_training,
                      add_v1net=model_config.add_v1net,
                  #  v1_timesteps=model_config.v1_timesteps,
                  #  v1_kernel_size=model_config.v1_kernel_size,
                   )
    model_vars = [var for var in tf.global_variables()]
    self.model_vars = set(model_vars).difference(set(all_vars))
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
    restore_vars = [var for var in self.model_vars 
                    if var.name[:-2] in restore_vars]
    saver = tf.train.Saver(restore_vars)
    saver.restore(sess, latest_checkpoint)