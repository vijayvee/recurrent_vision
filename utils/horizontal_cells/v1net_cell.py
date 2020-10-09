import tensorflow as tf
from tensorflow.python.framework import tensor_shape  # pylint: disable=import-error
from tensorflow.python.ops import array_ops  # pylint: disable=import-error
from tensorflow.python.ops import init_ops  # pylint: disable=import-error
from tensorflow.python.ops import math_ops  # pylint: disable=import-error
from tensorflow.python.ops import nn_ops  # pylint: disable=import-error
from tensorflow.python.ops import rnn_cell_impl  # pylint: disable=import-error
from tensorflow.python.ops import variable_scope as vs  # pylint: disable=import-error


class V1Net_BN_cell(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self,
               input_shape,
               output_channels,
               kernel_shape,
               inh_mult,
               exc_mult,
               activation=None,
               use_bias=True,
               timesteps=None,
               pointwise=False,
               forget_bias=1.0,
               initializers=None,
               training=None,
               name="v1net_cell"):
    """Construct V1net cell.
    Args:
      input_shape: Shape of the input as int tuple, excluding the batch size.
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
      initializers: Unused.
      training: Boolean training flag
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(V1Net_BN_cell, self).__init__(name=name)
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._inh_mult = inh_mult
    self._exc_mult = exc_mult
    self._use_bias = use_bias
    self.pointwise = pointwise
    self.activation = activation
    print('Setting training to', training)
    self.training = training
    self.initializers = initializers
    self.timesteps = timesteps
    self._total_output_channels = output_channels

    state_size = tensor_shape.TensorShape(
      self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
      self._input_shape[:-1] + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    activation_fn = get_activation(self.activation)
    cell, hidden = state
    new_hidden = _separable_conv([inputs, hidden],
                                  self._kernel_shape,
                                  self._output_channels,
                                  self._use_bias,
                                  inh_mult=self._inh_mult,
                                  exc_mult=self._exc_mult,
                                  pointwise=self.pointwise,
                                  activation=self.activation,
                                  initializers=self.initializers,
                                  dtype=inputs.dtype
                                  )
    conv_x, conv_h, conv_exc, conv_inh, conv_shunt = new_hidden
    X_i, X_f, X_c, X_o = array_ops.split(conv_x,
                                         num_or_size_splits=4,
                                         axis=-1,
                                         )
    H_i, H_f, H_o = array_ops.split(conv_h,
                                    num_or_size_splits=3,
                                    axis=-1,
                                    )
    H_exc, H_inh, H_shunt = conv_exc, conv_inh, conv_shunt
    
    # computing the gates
    input_gate = tf.math.add(X_i, H_i, name='input_gate')
    forget_gate = tf.math.add(X_f, H_f, name='forget_gate')
    output_gate = tf.math.add(X_o, H_o, name='output_gate')

    # computing horizontal push/pull
    input_hor, hidden_hor = X_c, (H_exc, H_inh, H_shunt)
    new_input = _horizontal(input_hor, hidden_hor)
    new_cell = math_ops.sigmoid(forget_gate) * cell
    new_cell += math_ops.sigmoid(input_gate) * tf.nn.tanh(new_input)
    new_cell = tf.keras.layers.LayerNormalization()(new_cell)  # (new_cell)
    if activation_fn:
      output = activation_fn(new_cell) * math_ops.sigmoid(output_gate)
    else:
      output = new_cell * math_ops.sigmoid(output_gate)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state


def get_activation(activation):
  """Function that returns an instance
  of an activation function"""
  if activation == 'relu':
    return tf.nn.relu
  if activation == 'sigmoid':
    return tf.nn.sigmoid
  if activation == 'tanh':
    return tf.nn.tanh


def _horizontal(input_hor, hidden_hor, control=False):
  """Function to perform hidden push pull
  integration in a linear-nonlinear fashion"""
  X_c, (H_exc, H_inh, H_shunt) = input_hor, hidden_hor
  # alpha = vs.get_variable(
  #                 "div_inh_control",
  #                 [k_in], initializer=get_initializer(dtype),
  #                 dtype=dtype)
  # beta = vs.get_variable(
  #                 "sub_inh_control",
  #                 [k_in], initializer=get_initializer(dtype),
  #                 dtype=dtype)
  context_mod = tf.nn.sigmoid(H_shunt) * (X_c + tf.nn.sigmoid(H_exc) - tf.nn.sigmoid(H_inh))  # divisive inhibition
  # context_mod = tf.nn.sigmoid(alpha) * tf.nn.relu(H_shunt) * (X_c + tf.nn.relu(H_exc) - tf.nn.sigmoid(beta) * tf.nn.relu(H_inh)) # controlled horizontal processing
  # context_mod = tf.nn.sigmoid(H_shunt) * (X_c + tf.nn.relu(H_exc) - tf.nn.relu(H_inh))  # divisive inhibition
  # context_mod = tf.nn.sigmoid(H_shunt) * (X_c + H_exc - H_inh)
  return context_mod


def get_initializer(dtype=tf.float32):
  initializer = tf.compat.v1.variance_scaling_initializer(
    seed=None,
    dtype=dtype,
  )
  return initializer


def _separable_conv(args, filter_size, output_channels, bias,
                    inh_mult=1.5, exc_mult=3, bias_start=0.0,
                    activation=None, initializers=None, pointwise=False,
                    channel_multiplier=1,
                    dtype=tf.float32):
  """Separable Convolution.
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    output_channels: int, number of convolutional kernels.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  # Calculate the total size of arguments on dimension 1.
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  n_args = len(args)
  input, hidden = args
  if n_args > 2:
    raise ValueError("Expected only two "
                     "arguments (input, hidden)")

  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Expected only 4-D arrays of "
                       "form [n,h,w,c] for performing 2D convolutions"
                       )
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
  x_arg_depth = shapes[0][-1]
  h_arg_depth = shapes[1][-1]
  separable_conv_op = tf.nn.separable_conv2d
  strides = shape_length * [1]
  filter_size_h = filter_size
  if pointwise:
    filter_size_h = [1, 1]
    print('Pointwise gates added')
  f_h, f_w = filter_size
  f_h_inh, f_w_inh = int(f_h * inh_mult), int(f_w * inh_mult)
  f_h_exc, f_w_exc = int(f_h * exc_mult), int(f_w * exc_mult)
  filter_size_inh = [f_h_inh, f_w_inh]
  filter_size_exc = [f_h_exc, f_w_exc]
  # Build input and hidden kernels
  x_kernel = vs.get_variable(
    "input_kernel", filter_size + [x_arg_depth, channel_multiplier],
    initializer=get_initializer(dtype),
    dtype=dtype)

  x_kernel_ps = vs.get_variable(
    "input_kernel_ps", [1, 1, channel_multiplier * x_arg_depth, output_channels * 4],
    initializer=get_initializer(dtype),
    dtype=dtype)

  # Build hidden state kernels
  h_kernel_gates = vs.get_variable(
    "hidden_kernel_g", filter_size_h + [h_arg_depth, channel_multiplier],
    initializer=get_initializer(dtype),
    dtype=dtype)

  h_kernel_gates_ps = vs.get_variable(
    "hidden_kernel_g_ps", [1, 1, channel_multiplier * h_arg_depth, output_channels * 3],
    initializer=get_initializer(dtype),
    dtype=dtype)

  # TODO: find optimal l1 strength
  h_kernel_inh = vs.get_variable(
    "hidden_kernel_inh", filter_size_inh + [h_arg_depth, channel_multiplier],
    initializer=get_initializer(dtype),
    dtype=dtype)
  h_kernel_inh_ps = vs.get_variable(
    "hidden_kernel_inh_ps", [1, 1, h_arg_depth * channel_multiplier, output_channels],
    initializer=get_initializer(dtype),
    dtype=dtype)

  h_kernel_shunt = vs.get_variable(
    "hidden_kernel_shunt", filter_size_inh + [h_arg_depth, channel_multiplier],
    initializer=get_initializer(dtype),
    dtype=dtype)
  h_kernel_shunt_ps = vs.get_variable(
    "hidden_kernel_shunt_ps", [1, 1, h_arg_depth * channel_multiplier, output_channels],
    initializer=get_initializer(dtype),
    dtype=dtype)

  h_kernel_exc = vs.get_variable(
    "hidden_kernel_exc", filter_size_exc + [h_arg_depth, channel_multiplier],
    initializer=get_initializer(dtype),
    dtype=dtype)
  h_kernel_exc_ps = vs.get_variable(
    "hidden_kernel_exc_ps", [1, 1, h_arg_depth * channel_multiplier, output_channels],
    initializer=get_initializer(dtype),
    dtype=dtype)
  res_x = separable_conv_op(input,
                            x_kernel,
                            x_kernel_ps,
                            strides,
                            padding="SAME")

  res_h_gates = separable_conv_op(hidden,
                                  h_kernel_gates,
                                  h_kernel_gates_ps,
                                  strides,
                                  padding="SAME")
  res_h_inh = separable_conv_op(hidden,
                                h_kernel_inh,
                                h_kernel_inh_ps,
                                strides,
                                padding="SAME")
  res_h_exc = separable_conv_op(hidden,
                                h_kernel_exc,
                                h_kernel_exc_ps,
                                strides,
                                padding="SAME")
  res_h_shunt = separable_conv_op(hidden,
                                  h_kernel_shunt,
                                  h_kernel_shunt_ps,
                                  strides,
                                  padding="SAME")
  bias_input = vs.get_variable(
    "biases_input", [output_channels * 4],
    dtype=dtype,
    initializer=init_ops.constant_initializer(
      bias_start,
      dtype=dtype))
  bias_hidden_gates = vs.get_variable(
    "biases_hidden_g", [output_channels * 3],
    dtype=dtype,
    initializer=init_ops.constant_initializer(
      bias_start,
      dtype=dtype))
  bias_hidden_exc = vs.get_variable(
    "biases_hidden_e", [output_channels],
    dtype=dtype,
    initializer=init_ops.constant_initializer(
      bias_start,
      dtype=dtype))
  bias_hidden_inh = vs.get_variable(
    "biases_hidden_i", [output_channels],
    dtype=dtype,
    initializer=init_ops.constant_initializer(
      bias_start,
      dtype=dtype))
  bias_hidden_shunt = vs.get_variable(
    "biases_hidden_shunt", [output_channels],
    dtype=dtype,
    initializer=init_ops.constant_initializer(
      bias_start,
      dtype=dtype))
  res_input = tf.math.add(res_x,
                          bias_input,
                          name='conv_input_gates'
                          )
  res_hidden_gates = tf.math.add(res_h_gates,
                                 bias_hidden_gates,
                                 name='conv_hidden_gates'
                                 )
  res_hidden_inh = tf.math.add(res_h_inh,
                               bias_hidden_inh,
                               name='conv_hidden_inh'
                               )
  res_hidden_shunt = tf.math.add(res_h_shunt,
                                 bias_hidden_shunt,
                                 name='conv_hidden_shunt'
                                 )
  res_hidden_exc = tf.math.add(res_h_exc,
                               bias_hidden_exc,
                               name='conv_hidden_exc'
                               )
  return (res_input, res_hidden_gates,
          res_hidden_exc, res_hidden_inh,
          res_hidden_shunt)
