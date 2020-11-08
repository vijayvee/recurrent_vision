"""Create optimization functions."""
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
tf.disable_v2_behavior()


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



def get_optimizer(learning_rate, opt, use_tpu=True):
  """Function to return optimizer instance."""
  if opt == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  elif opt == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
  elif opt == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif opt == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9,
                                           use_nesterov=True)
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
  return optimizer
