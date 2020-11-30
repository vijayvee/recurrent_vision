"""Utility functions to build model layers"""

import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import tf_slim as slim   # pylint: disable=import-error
from absl import flags
from recurrent_vision.utils.horizontal_cells.v1net_cell import V1Net_BN_cell
from recurrent_vision.utils.horizontal_cells.v1net_cell_fn import V1Net_functional_cell
from recurrent_vision.utils.horizontal_cells.v1net_compact_cell import V1NetCompact
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

def fuse_predictions(inputs, is_training=True):
  """Build 1x1 convolution for side output fusion (HED)."""
  initializer = tf.keras.initializers.Constant(value=1./5)
  inputs = tf.layers.conv2d(inputs=inputs,
                            filters=1,
                            kernel_size=1,
                            strides=(1,1),
                            padding='same',
                            kernel_initializer=initializer,
                            )
  return inputs

def build_conv_bn_relu(inputs,
                       filters, 
                       kernel_size, 
                       strides=(1, 1), 
                       padding='same',
                       is_training=True,
                       normalization=True,
                       activation=True,
                       ):
  """Function to build a Conv+Batchnorm+Relu block.
  Args:
    inputs: Tensor of input images 
    filters: Integer number of filters
    kernel_size: Integer spatial size of kernels
    strides: Tuple of (horizontal, vertical) conv stride
    padding: 'same' or 'valid' indicating padding style
    is_training: Boolean indicating whether model is in training mode

  Returns:
    Tensor of Relu(BatchNorm(Conv2D(images)))
  """
  inputs = tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            )
  if normalization:
    inputs = tf.layers.batch_normalization(inputs=inputs,
                                         axis=3,
                                         center=True,
                                         scale=True,
                                         training=is_training,
                                         gamma_initializer=tf.ones_initializer())
  if activation:
    inputs = tf.nn.relu(inputs)
  return inputs

def build_pool2d(inputs,
                 pool_size=2,
                 strides=(2,2),
                 ):
  """Function to build spatial max pooling.
  Args:
    inputs: Tensor of input images
    pool_size: Integer size of pooling window
    strides: Tuple of (horizontal, vertical) conv stride.

  Returns:
    Max pooling applied input tensor
  """
  inputs = tf.layers.max_pooling2d(inputs=inputs,
                                   pool_size=pool_size,
                                   strides=strides)
  return inputs

def build_avgpool(inputs):
  """Function to build global average pooling.
  Args:
    inputs: Tensor of input images
    pool_size: Integer size of pooling window
    strides: Tuple of (horizontal, vertical) conv stride.

  Returns:
    Average pooling applied tensor
  """
  # TODO(vveeraba): Add test for global average pool
  _,h,w,_ = inputs.shape.as_list()
  inputs = tf.layers.average_pooling2d(inputs=inputs,
                                       pool_size=h,
                                       strides=(h,w))
  return inputs

def build_v1net_new(inputs, timesteps,
                filters, kernel_size,
                is_training=True, inh_mult=1.5,
                exc_mult=3, v1_act='relu',
                compact=False):
  """Build V1Net layer (functional impl).
  Args:
    inputs: Input tensor (n,h,w,c)
    timesteps: Integer Number of recurrent timesteps
    filters: Integer Number of filters
    kernel_size: Integer Spatial convolution size
    is_training: Boolean training flag
    inh_mult: Float multiplier for 
              inhibitory spatial convolution size
    exc_mult: Float multiplier for 
              excitatory spatial convolution size
    v1_act: String activation function for v1net's output
  Returns:
    Applying V1Net layer to inputs
  """
  n, h, w, _ = inputs.shape.as_list()
  ones = tf.ones([timesteps, 1, 1, 1, 1], 
                 dtype=tf.float32)
  v1net_input = ones * inputs
  v1net_input = tf.transpose(v1net_input, [1, 0, 2, 3, 4])
  state_c = tf.zeros([n, h, w, filters], 
                     dtype=inputs.dtype, 
                     name='init_c')
  state_h = tf.zeros([n, h, w, filters], 
                     dtype=inputs.dtype, 
                     name='init_h')
  state = tf.nn.rnn_cell.LSTMStateTuple(state_c, 
                                        state_h)
  cell = V1Net_functional_cell
  tf.logging.info("Using functional implementation of v1net")
  v1net_cell = cell(output_channels=filters,
                    kernel_size=kernel_size,
                    inh_mult=inh_mult,
                    exc_mult=exc_mult,
                    activation=v1_act,
                    timesteps=timesteps,
                    training=is_training,
                    )
  _, new_state = v1net_cell.build_v1net(inputs, state)
  _, new_state_h = new_state
  return new_state_h

def build_v1net(inputs, timesteps,
                filters, kernel_size,
                is_training=True, inh_mult=1.5,
                exc_mult=3, v1_act='relu',
                compact=False):
  """Build V1Net layer.
  Args:
    inputs: Input tensor (n,h,w,c)
    timesteps: Integer Number of recurrent timesteps
    filters: Integer Number of filters
    kernel_size: Integer Spatial convolution size
    is_training: Boolean training flag
    inh_mult: Float multiplier for 
              inhibitory spatial convolution size
    exc_mult: Float multiplier for 
              excitatory spatial convolution size
    v1_act: String activation function for v1net's output
  Returns:
    Applying V1Net layer to inputs
  """
  n, h, w, c = inputs.shape.as_list()
  ones = tf.ones([timesteps, 1, 1, 1, 1], 
                 dtype=tf.float32)
  v1net_input = ones * inputs
  v1net_input = tf.transpose(v1net_input, [1, 0, 2, 3, 4])
  state_c = tf.zeros([n, h, w, filters], 
                     dtype=inputs.dtype, 
                     name='init_c')
  state_h = tf.zeros([n, h, w, filters], 
                     dtype=inputs.dtype, 
                     name='init_h')
  state = tf.nn.rnn_cell.LSTMStateTuple(state_c, 
                                        state_h)
  if compact:
    cell = V1NetCompact
  else:
    cell = V1Net_BN_cell
  v1net_cell = cell(input_shape=[h, w, c],
                    output_channels=filters,
                    kernel_shape=[kernel_size, 
                                  kernel_size],
                    inh_mult=inh_mult,
                    exc_mult=exc_mult,
                    activation=v1_act,
                    timesteps=timesteps,
                    training=is_training,
                    )
  v1net_out = tf.nn.dynamic_rnn(cell=v1net_cell,
                                inputs=v1net_input,
                                initial_state=state,
                                dtype=tf.float32,
                                )
  _, new_state_h = v1net_out[1]
  return new_state_h

def add_v1net_layer(net, is_training=True, 
                    add_v1net=True, 
                    v1net_idx=0):
  """Function to add v1net layer."""
  if add_v1net and FLAGS.v1_timesteps:
    with tf.variable_scope("v1net-conv%s" % v1net_idx):
      n_filters = int(net.shape.as_list()[-1])
      v1_timesteps, v1_kernel_size = FLAGS.v1_timesteps, 3
      net = build_v1net(inputs=net, 
                        filters=n_filters, 
                        timesteps=v1_timesteps, 
                        kernel_size=v1_kernel_size,
                        is_training=is_training)
  return net

def build_dense(inputs,
                units):
  """Function to build dense layer.
  Args:
    inputs: Tensor of input activations
    units: Integer number of output units
    
  Returns:
    Output of dense layer
  """
  inputs = tf.layers.dense(inputs=inputs,
                           units=units)
  return inputs

def build_hed_output(side_outputs, 
                     height, width,
                     reuse=None,
                     reduce_conv=True):
  """Function to build HED boundary prediction output.
  Args:
    inputs: Tensor of input activations    
  Returns:
    Output of dense layer
  """
  side_outputs_fullres = [tf.image.resize_bilinear(side_output, [height,width])
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
  side_outputs_fullres = tf.reshape(side_outputs_fullres,
                                    (-1, height, width, 1))
  return fused_predictions, side_outputs_fullres

def get_upsampling_weight(in_channels=1, out_channels=1, kernel_size=4):
  """Make a 2D bilinear kernel suitable for upsampling"""
  factor = (kernel_size + 1) // 2
  if kernel_size % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5
  og = np.ogrid[:kernel_size, :kernel_size]
  filt = (1 - abs(og[0] - center) / factor) * \
          (1 - abs(og[1] - center) / factor)
  weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                    dtype=np.float64)
  weight[range(in_channels), range(out_channels), :, :] = filt
  weight = np.transpose(weight, (2, 3, 0, 1))
  return np.float32(weight)

def resize_and_crop(net, scale, height, width):
  """Function to resize with conv2d_transpose 
  and crop to [height, width] dimensions."""
  weights_upsample = get_upsampling_weight(kernel_size=scale*2)
  weights_init = tf.constant_initializer(value=weights_upsample)
  net = slim.conv2d_transpose(net, 1, scale*2, stride=scale,
                              weights_initializer=weights_init,
                              biases_initializer=tf.constant_initializer(0.),
                              activation_fn=None, normalizer_fn=None,
                              trainable=False)
  _, h, w, _ = net.shape.as_list()
  assert (h, w) == (height, width)
  return net