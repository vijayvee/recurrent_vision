import tensorflow.compat.v1 as tf  # pylint: disable=import-error
from tensorflow.python.framework import tensor_shape  # pylint: disable=import-error
from tensorflow.python.ops import init_ops  # pylint: disable=import-error
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
               control=False,
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
    self.control = control
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
    conv_xh, conv_exc, conv_shunt = new_hidden
    xh_tensors = tf.split(conv_xh,
                          num_or_size_splits=4,
                          axis=-1,
                          )
    input_gate, forget_gate, input_hor, output_gate = xh_tensors
    hidden_hor = (conv_exc, conv_shunt)
    
    new_input = _horizontal(input_hor, hidden_hor, self.control)
    new_cell = tf.nn.sigmoid(forget_gate) * cell
    new_cell += tf.nn.sigmoid(input_gate) * tf.nn.tanh(new_input)
    new_cell = tf.keras.layers.LayerNormalization()(new_cell)
    if activation_fn:
      output = activation_fn(new_cell) * tf.nn.sigmoid(output_gate)
    else:
      output = new_cell * tf.nn.sigmoid(output_gate)
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


def get_initializer(dtype=tf.float32):
  initializer = tf.compat.v1.variance_scaling_initializer(
    seed=None,
    dtype=dtype,
  )
  return initializer


def _separable_conv(args, filter_size, output_channels, bias,
                    inh_mult=1.5, exc_mult=3, bias_start=0.0,
                    activation=None, initializers=None, 
                    pointwise=False, channel_multiplier=1,
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

  input, hidden = args
  h_arg_depth = shapes[1][-1]
  x_h = tf.concat([input, hidden], axis=-1)
  xh_arg_depth = x_h.shape.as_list()[-1]

  separable_conv_op = tf.nn.separable_conv2d
  strides = shape_length * [1]

  if pointwise:
    filter_size = [1, 1]
    print('Pointwise gates added')

  f_h, f_w = filter_size
  f_h_inh, f_w_inh = int(f_h * inh_mult), int(f_w * inh_mult)
  f_h_exc, f_w_exc = int(f_h * exc_mult), int(f_w * exc_mult)
  filter_size_inh = [f_h_inh, f_w_inh]
  filter_size_exc = [f_h_exc, f_w_exc]

  # Build input and hidden kernels
  xh_kernel = vs.get_variable(
    "input_hidden_kernel", filter_size + [xh_arg_depth, channel_multiplier],
    initializer=get_initializer(dtype),
    dtype=dtype)

  xh_kernel_ps = vs.get_variable(
    "input_hidden_kernel_ps", [1, 1, channel_multiplier * xh_arg_depth, output_channels * 4],
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

  res_xh = separable_conv_op(x_h,
                             xh_kernel,
                             xh_kernel_ps,
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

  bias_xh = vs.get_variable(
    "biases_input_hidden", [output_channels * 4],
    dtype=dtype, initializer=init_ops.constant_initializer(
                                        bias_start,
                                        dtype=dtype))
  
  bias_hidden_exc = vs.get_variable(
    "biases_hidden_exc", [output_channels],
    dtype=dtype, initializer=init_ops.constant_initializer(
                                        bias_start,
                                        dtype=dtype))

  bias_hidden_shunt = vs.get_variable(
    "biases_hidden_shunt", [output_channels],
    dtype=dtype, initializer=init_ops.constant_initializer(
                                        bias_start,
                                        dtype=dtype))

  res_input_hidden = tf.math.add(res_xh,
                          bias_xh,
                          name='conv_input_hidden'
                          )

  res_hidden_exc = tf.math.add(res_h_exc,
                               bias_hidden_exc,
                               name='conv_hidden_exc'
                               )

  res_hidden_shunt = tf.math.add(res_h_shunt,
                                bias_hidden_shunt,
                                name='conv_hidden_shunt'
                                )

  return (res_input_hidden, res_hidden_exc, res_hidden_shunt)
