"""Create optimization functions."""
import numpy as np
import tensorflow as tf

def get_optimizer(learning_rate, opt, use_tpu=True):
  """Function to return optimizer instance."""
  if opt == 'adam':
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
  elif opt == 'rmsprop':
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
  elif opt == 'sgd':
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif opt == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9,
                                           use_nesterov=True)
  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
  return optimizer