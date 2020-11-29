"""Create optimization functions."""
from absl import flags
import re
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

def filter_variables(variables, filter_regex_list, invert=True):
  """Filters out the variables matching the filter_regex.
  Filter out the variables whose name matches the any of the regular
  expressions in filter_regex_list and returns the remaining variables.
  Optionally, if invert=True, the complement set is returned.
  Args:
    variables: a list of tensorflow variables.
    filter_regex_list: a list of string regular expressions.
    invert: (boolean).  If True, returns the complement of the filter set; that
      is, all variables matching filter_regex are kept and all others discarded.
  Returns:
    a list of filtered variables.
  """
  kept_vars = []
  variables_to_ignore_patterns = list([fre for fre in filter_regex_list if fre])
  for var in variables:
    add = True
    for pattern in variables_to_ignore_patterns:
      if re.match(pattern, var.op.name):
        add = False
        break
    if add != invert:
      kept_vars.append(var)
  return kept_vars

def build_learning_rate(initial_lr,
                        global_step,
                        steps_per_epoch=None,
                        lr_decay_type='exponential',
                        decay_factor=0.1,
                        decay_steps=20000,
                        total_steps=None,
                        warmup_epochs=None):
  """Build learning rate."""
  if lr_decay_type == 'exponential':
    assert steps_per_epoch is not None
    lr = tf.train.exponential_decay(
        initial_lr, global_step, decay_steps, 
        decay_factor, staircase=True)
  elif lr_decay_type == 'cosine':
    assert total_steps is not None
    lr = 0.5 * initial_lr * (
        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
  elif lr_decay_type == 'constant':
    lr = initial_lr
  else:
    assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

  if warmup_epochs:
    tf.logging.info('Learning rate warmup_epochs: %d', warmup_epochs)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_lr = (
        initial_lr * tf.cast(global_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  return lr

def get_optimizer(loss, learning_rate, vars=None, opt=None, use_tpu=True):
  """Function to return optimizer instance."""
  global_step = tf.train.get_global_step()
  train_op = None
  if opt == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  elif opt == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  elif opt == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif opt == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9,
                                           use_nesterov=False)
  elif opt == 'bsds_momentum':
    del vars  # unused here
    train_op = get_optimizer_bsds(loss, learning_rate, use_tpu)

  if not train_op:
    if use_tpu:
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step, vars)
  return train_op


def get_optimizer_bsds(loss, learning_rate, use_tpu=True):
  """Function to load a BSDS-HED optimizer."""
  lr_mult = {}
  vars = list(tf.trainable_variables())
  early_weights = filter_variables(vars, ["vgg_16/conv[1-4]/conv[1-4]_[1-3]/weights"])
  early_biases = filter_variables(vars, ["vgg_16/conv[1-4]/conv[1-4]_[1-3]/biases"])
  late_weights = filter_variables(vars, ["vgg_16/conv5/conv5_[1-3]/weights"])
  late_biases = filter_variables(vars, ["vgg_16/conv5/conv5_[1-3]/biases"])
  dsn_weights = filter_variables(vars, ["vgg_16/dsn_convolution_[1-5]/Conv/weights"])
  dsn_biases = filter_variables(vars, ["vgg_16/dsn_convolution_[1-5]/Conv/biases"])
  fusion_weights = filter_variables(vars, ["vgg_16/side_output_fusion/Conv/weights"])
  fusion_biases = filter_variables(vars, ["vgg_16/side_output_fusion/Conv/biases"])
  # TODO(vveeraba): Add v1net optimization here

  for w_var, b_var in zip(early_weights, early_biases):
    lr_mult[w_var.name] = 1
    lr_mult[b_var.name] = 2

  for w_var, b_var in zip(late_weights, late_biases):
    lr_mult[w_var.name] = 100
    lr_mult[b_var.name] = 200
    
  for w_var, b_var in zip(dsn_weights, dsn_biases):
    lr_mult[w_var.name] = 0.01
    lr_mult[b_var.name] = 0.02

  for w_var, b_var in zip(fusion_weights, fusion_biases):
    lr_mult[w_var.name] = 0.001
    lr_mult[b_var.name] = 0.002

  optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                         momentum=0.9,
                                         use_nesterov=False)
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  grads_vars = optimizer.compute_gradients(loss)
  grads_vars_mult = []
  for grad, var in grads_vars:
    grad = grad * lr_mult[var.name]
    grads_vars_mult.append((grad, var))
  train_op = optimizer.apply_gradients(grads_vars_mult)
  return train_op
