"""Functional implementation of V1Net."""
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
from tensorflow.python.framework import tensor_shape  # pylint: disable=import-error
from tensorflow.python.ops import init_ops  # pylint: disable=import-error
from tensorflow.python.ops import rnn_cell_impl  # pylint: disable=import-error
from tensorflow.python.ops import variable_scope as vs  # pylint: disable=import-error

tf.disable_v2_behavior()

def get_activation(activation):
  """Function that returns an instance
  of an activation function"""
  if activation == 'relu':
    return tf.nn.relu
  if activation == 'sigmoid':
    return tf.nn.sigmoid
  if activation == 'tanh':
    return tf.nn.tanh

def get_initializer(dtype=tf.float32):
  initializer = tf.compat.v1.variance_scaling_initializer(
    seed=None,
    dtype=dtype,
  )
  return initializer

def _horizontal(input_hor, hidden_hor, control=False):
  """Function to perform hidden push pull
  integration in a linear-nonlinear fashion"""
  X_c, (H_exc, H_shunt) = input_hor, hidden_hor
  k_in = X_c.shape.as_list()[-1]
  dtype = X_c.dtype
  if control:
    # controlled horizontal processing
    with tf.variable_scope("horizontal_processing",
                           reuse=tf.AUTO_REUSE):
      alpha = vs.get_variable(
                   "div_inh_control",
                   [k_in], initializer=get_initializer(dtype),
                   dtype=dtype)
      beta = vs.get_variable(
                   "exc_control",
                   [k_in], initializer=get_initializer(dtype),
                   dtype=dtype)
      context_mod = tf.nn.sigmoid(alpha) * tf.nn.relu(H_shunt) * (X_c + tf.nn.sigmoid(beta) * tf.nn.relu(H_exc)) 
  else:
    with tf.variable_scope("horizontal_processing",
                           reuse=tf.AUTO_REUSE):
      context_mod = tf.nn.sigmoid(H_shunt) * (X_c + tf.nn.sigmoid(H_exc))
  return context_mod


class V1Net_functional_cell(object):
  """Convolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """
  def __init__(self,
               output_channels,
               kernel_size,
               inh_mult=1,
               exc_mult=1,
               activation=None,
               use_bias=True,
               timesteps=1,
               pointwise=False,
               forget_bias=1.0,
               training=None,
               control=False,
               name="v1net_cell"):
    """Construct V1net cell.
    Args:
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as an int tuple (of size 1, 2 or 3).
      inh_mult: Float multiplier for 
              inhibitory spatial convolution size
      exc_mult: Float multiplier for 
              excitatory spatial convolution size
      activation: String activation function for v1net's output
      use_bias: (bool) Use bias in convolutions.
      timesteps: Integer Number of recurrent timesteps
      forget_bias: Forget bias.
      training: Boolean training flag
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    self._output_channels = output_channels
    self._kernel_size = int(kernel_size)
    self._inh_kernel_size = int(kernel_size * inh_mult)
    self._exc_kernel_size = int(kernel_size * exc_mult)
    self._use_bias = use_bias
    self.pointwise = pointwise
    self.activation = activation
    self.training = training
    self.control = control
    self.timesteps = timesteps
    self._total_output_channels = output_channels

  def build_v1net(self, inputs, state):
    """Function to build v1net ops."""
    for _ in range(self.timesteps):
      net, state = self.step(inputs, state)
    return net, state
  
  def step(self, inputs, state):
    """Function to implement one step of v1net processing."""
    activation_fn = get_activation(self.activation)
    cell, hidden = state
    x_h = tf.concat([inputs, hidden], axis=-1)
    vs_init = get_initializer(inputs.dtype)
    with tf.variable_scope(tf.get_variable_scope(),
                           reuse=tf.AUTO_REUSE):
      with tf.variable_scope("x_h_convolution",
                           reuse=tf.AUTO_REUSE):
        conv_xh = tf.keras.layers.SeparableConv2D(
                      filters=self._output_channels*4,
                      kernel_size=self._kernel_size,
                      strides=(1,1), padding='same',
                      use_bias=True, depthwise_initializer=vs_init, 
                      pointwise_initializer=vs_init,
                      bias_initializer='zeros', 
                      )(x_h)
      with tf.variable_scope("exc_convolution",
                           reuse=tf.AUTO_REUSE):
        conv_exc = tf.keras.layers.SeparableConv2D(
                      filters=self._output_channels,
                      kernel_size=self._exc_kernel_size,
                      strides=(1,1), padding='same',
                      use_bias=True, depthwise_initializer=vs_init, 
                      pointwise_initializer=vs_init,
                      bias_initializer='zeros', 
                      )(hidden)
      with tf.variable_scope("shunt_convolution",
                           reuse=tf.AUTO_REUSE):
        conv_shunt = tf.keras.layers.SeparableConv2D(
                      filters=self._output_channels,
                      kernel_size=self._inh_kernel_size,
                      strides=(1,1), padding='same',
                      use_bias=True, depthwise_initializer=vs_init, 
                      pointwise_initializer=vs_init,
                      bias_initializer='zeros', 
                      )(hidden)
    gates = tf.split(conv_xh,
                      num_or_size_splits=4,
                      axis=-1,
                      )
    input_gate, forget_gate, input_hor, output_gate = gates
    new_input = _horizontal(input_hor, (conv_exc, conv_shunt), self.control)
    new_cell = tf.nn.sigmoid(forget_gate) * cell
    new_cell += tf.nn.sigmoid(input_gate) * tf.nn.tanh(new_input)
    new_cell = tf.keras.layers.LayerNormalization()(new_cell)
    if activation_fn:
      output = activation_fn(new_cell) * tf.nn.sigmoid(output_gate)
    else:
      output = new_cell * tf.nn.sigmoid(output_gate)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state