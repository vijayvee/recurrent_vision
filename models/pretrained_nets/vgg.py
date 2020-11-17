# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import, division, print_function

import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import tf_slim as slim  # pylint: disable=import-error

from absl import flags
from recurrent_vision.utils.model_utils import build_v1net, fuse_predictions

FLAGS = flags.FLAGS

def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      # weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          add_v1net_early=False,
          add_v1net=False,
          reuse=None,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_a', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      if add_v1net and FLAGS.v1_timesteps:
        v1_timesteps, v1_kernel_size = 6, 5
        tf.logging.INFO("Adding V1Net with %s timesteps, %s kernel_size" % (v1_timesteps,
                                                                            v1_kernel_size))
        net = build_v1net(inputs=net, filters=64, 
                          timesteps=v1_timesteps, 
                          kernel_size=v1_kernel_size)
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(
            input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224


def vgg_16(inputs, cams=None,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           add_v1net_early=False,
           add_v1net=False,
           reuse=None,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  del cams  # unused here
  with tf.variable_scope(
      scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      if add_v1net_early and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv1"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 64
          net = build_v1net(inputs=net, filters=n_filters,
                            timesteps=v1_timesteps,
                            kernel_size=v1_kernel_size)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')

      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv2"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 128
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv3"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 256
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')

      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv4"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 512
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')

      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv5"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 512
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(
            input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224


def vgg_19(inputs, cams=None,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           add_v1net_early=False,
           add_v1net=False,
           reuse=None,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  del cams  # unused here
  with tf.variable_scope(
      scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      if add_v1net and FLAGS.v1_timesteps:
        v1_timesteps, v1_kernel_size = 6, 5
        tf.logging.INFO("Adding V1Net with %s timesteps, %s kernel_size" % (v1_timesteps,
                                                                            v1_kernel_size))
        net = build_v1net(inputs=net, filters=64, 
                          timesteps=v1_timesteps, 
                          kernel_size=v1_kernel_size)
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(
            input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224


def vgg_16_hed(inputs, cams=None,
              num_classes=1,
              is_training=True,
              add_v1net_early=False,
              add_v1net=False,
              reuse=None,
              reduce_conv=True,
              scope='vgg_16',
              ):
  """VGG-16 implementation of HED.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether or not the model is being trained.
    add_v1net: whether to add v1net blocks after convolutions.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
  Returns:
    side_outputs_fullres: list of side output logits resized to input resolution.
    end_points: a dict of tensors with intermediate activations.
  """
  del cams  # unused here
  side_outputs = []
  _, h, w, _ = inputs.shape.as_list()
  with tf.variable_scope(
      scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    # TODO(vveerabadran): Where to add V1Net?
    # TODO(vveerabadran): Should side outputs be output of V1Net?
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      # side_outputs.append(net)
      # net = slim.max_pool2d(net, [2, 2], scope='pool1')
      if add_v1net_early and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv1"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 64
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size,
                            is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')

      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv2"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 128
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size,
                            is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv3"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 256
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size,
                            is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv4"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 512
          net = build_v1net(inputs=net, filters=n_filters, 
                           timesteps=v1_timesteps, 
                           kernel_size=v1_kernel_size,
                           is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv5"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 512
          net = build_v1net(inputs=net, filters=n_filters, 
                           timesteps=v1_timesteps, 
                           kernel_size=v1_kernel_size,
                           is_training=is_training)
      side_outputs.append(net)
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      side_outputs_fullres = [tf.image.resize_bilinear(side_output, [h,w])
                              for side_output in side_outputs]
      with tf.variable_scope("side_output_classifiers", reuse=reuse):
        side_outputs_fullres = [slim.conv2d(side_output, 1, [1, 1],
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    )
                                for side_output in side_outputs_fullres]
      side_outputs_fullres = tf.stack(side_outputs_fullres, axis=0)
      if reduce_conv:
        with tf.variable_scope("side_output_fusion"):
          side_outputs_ = tf.transpose(side_outputs_fullres, 
                                            (1,2,3,4,0))
          side_outputs_ = tf.squeeze(side_outputs_, axis=3)
          fused_predictions = fuse_predictions(side_outputs_)
      else:
        fused_predictions = tf.reduce_mean(side_outputs_fullres, axis=0)
      end_points['fused_predictions'] = fused_predictions
      side_outputs_fullres = tf.reshape(side_outputs_fullres,
                                        (-1, h, w, 1))
      end_points['side_outputs_fullres'] = side_outputs_fullres
      return fused_predictions, end_points

def vgg_16_hed_cam(inputs, cams,
                    num_classes=1,
                    is_training=True,
                    add_v1net_early=False,
                    add_v1net=False,
                    reuse=None,
                    reduce_conv=True,
                    scope='vgg_16',
                    ):
  """VGG-16 implementation of HED.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether or not the model is being trained.
    add_v1net: whether to add v1net blocks after convolutions.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional scope for the variables.
  Returns:
    side_outputs_fullres: list of side output logits resized to input resolution.
    end_points: a dict of tensors with intermediate activations.
  """
  side_outputs = []
  _, h, w, _ = inputs.shape.as_list()
  with tf.variable_scope(
      scope, 'vgg_16', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    # TODO(vveerabadran): Where to add V1Net?
    # TODO(vveerabadran): Should side outputs be output of V1Net?
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      with tf.variable_scope("cam-conv1"):
        cam_net = slim.repeat(cams, 1, slim.conv2d, 64, [1, 1], scope="cam-conv1")
        net = net + cam_net

      if add_v1net_early and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv1"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 64
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size,
                            is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      cam_net = slim.max_pool2d(cam_net, [2, 2], scope='cam_pool1')

      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      with tf.variable_scope("cam-conv2"):
        cam_net = slim.repeat(cam_net, 1, slim.conv2d, 128, [1, 1], scope="cam-conv2")
        net = net + cam_net

      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv2"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 128
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size,
                            is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      cam_net = slim.max_pool2d(cam_net, [2, 2], scope='cam_pool2')

      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      with tf.variable_scope("cam-conv3"):
        cam_net = slim.repeat(cam_net, 1, slim.conv2d, 256, [1, 1], scope="cam-conv3")
        net = net + cam_net

      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv3"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 256
          net = build_v1net(inputs=net, filters=n_filters, 
                            timesteps=v1_timesteps, 
                            kernel_size=v1_kernel_size,
                            is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      cam_net = slim.max_pool2d(cam_net, [2, 2], scope='cam_pool3')

      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      with tf.variable_scope("cam-conv4"):
        cam_net = slim.repeat(cam_net, 1, slim.conv2d, 512, [1, 1], scope="cam-conv4")
        net = net + cam_net

      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv4"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 512
          net = build_v1net(inputs=net, filters=n_filters, 
                           timesteps=v1_timesteps, 
                           kernel_size=v1_kernel_size,
                           is_training=is_training)
      side_outputs.append(net)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      cam_net = slim.max_pool2d(cam_net, [2, 2], scope='cam_pool4')

      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      with tf.variable_scope("cam-conv5"):
        cam_net = slim.repeat(cam_net, 1, slim.conv2d, 512, [1, 1], scope="cam-conv5")
        net = net + cam_net

      if add_v1net and FLAGS.v1_timesteps:
        with tf.variable_scope("v1net-conv5"):
          v1_timesteps, v1_kernel_size, n_filters = FLAGS.v1_timesteps, 3, 512
          net = build_v1net(inputs=net, filters=n_filters, 
                           timesteps=v1_timesteps, 
                           kernel_size=v1_kernel_size,
                           is_training=is_training)
      side_outputs.append(net)
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      side_outputs_fullres = [tf.image.resize_bilinear(side_output, [h,w])
                              for side_output in side_outputs]
      with tf.variable_scope("side_output_classifiers", reuse=reuse):
        side_outputs_fullres = [slim.conv2d(side_output, 1, [1, 1],
                                    activation_fn=None,
                                    normalizer_fn=None,
                                    )
                                for side_output in side_outputs_fullres]
      side_outputs_fullres = tf.stack(side_outputs_fullres, axis=0)
      if reduce_conv:
        with tf.variable_scope("side_output_fusion"):
          side_outputs_ = tf.transpose(side_outputs_fullres, 
                                            (1,2,3,4,0))
          side_outputs_ = tf.squeeze(side_outputs_, axis=3)
          fused_predictions = fuse_predictions(side_outputs_)
      else:
        fused_predictions = tf.reduce_mean(side_outputs_fullres, axis=0)
      end_points['fused_predictions'] = fused_predictions
      side_outputs_fullres = tf.reshape(side_outputs_fullres,
                                        (-1, h, w, 1))
      end_points['side_outputs_fullres'] = side_outputs_fullres
      return fused_predictions, end_points


# Alias
vgg_d = vgg_16
vgg_e = vgg_19
