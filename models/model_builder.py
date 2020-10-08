"""Abstract class for building multi-layer models"""
from abc import ABC, abstractclassmethod

import numpy as np
import tensorflow as tf

from recurrent_vision.utils.model_utils import build_conv_bn_relu, build_pool2d

class ModelBuilder:
  """Abstract class for building models."""
  @abstractclassmethod
  def __init__(self, model_config):
    self.model_config = model_config

  @abstractclassmethod
  def preprocess(self, input_images):
    pass

  def augment_images(self, input_images):
    """Augment minibatch of images."""
    # TODO(vveeraba): Check the parameters of the following transformations
    print('Augmenting images..')
    input_shape = input_images.shape.as_list()
    input_images = tf.image.random_brightness(input_images,
                                              max_delta=0.3)
    input_images = tf.image.random_contrast(input_images,
                                            lower=0.4, upper=1.8)
    input_images = tf.image.random_hue(input_images, 0.08)
    input_images = tf.image.random_saturation(input_images, 0.5, 1.)
    input_images = tf.reshape(input_images, input_shape)
    return input_images

  @abstractclassmethod
  def build_model(self, input_images):
    pass

  @abstractclassmethod
  def restore_checkpoint(self, checkpoint_path):
    pass
  
  def convolution_stem(self, images, 
                       is_training=True):
    """Build a convolutional bottom stem."""
    images = build_conv_bn_relu(images,
                                filters=32,
                                kernel_size=7,
                                is_training=is_training)
    images = build_pool2d(images)
    return images