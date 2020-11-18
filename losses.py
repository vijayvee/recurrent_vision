"""Custom loss functions."""
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import numpy as np

tf.disable_v2_behavior()

def weighted_ce(logits, labels, lbda=1.1):
  """Weighted cross entropy.
  Args:
    logits: Tensor of predictions.
    labels: Tensor of ground truth (same shape as logits).
  Returns:
    weighted_cross_entropy(logits, labels)
  L = -beta*y*log(p) - (1-beta)*(1-y)*log(1-p)
    = -beta*y*log(1/1+exp(-z)) - (1-beta)*(1-y)log(exp(-z)/(1+exp(-z)))
    = beta*y*log(1+exp(-z)) - (1-beta)*(1-y)*(log(exp(-z)) - log(1+exp(-z)))
    = beta*y*q + z*(1-beta)*(1-y) + (1-beta)*(1-y)*q   [q=log(1+exp(-z))]
    = beta*y*q + (1-beta)*(1-y)*(z+q)
  """
  beta_true = tf.reduce_mean(labels, 
			     axis=(1, 2, 3), 
			     keepdims=True)  # num_true / num_true + num_false
  beta_false = 1. - beta_true
  q = tf.log1p(tf.exp(-logits))
  xentropy = lbda * beta_true * labels * q + beta_false * (1-labels) * (logits + q)
  xentropy = tf.reduce_mean(xentropy)
  return xentropy

def weighted_ce_bdcn(logits, labels, lbda=1.1, gamma=0.4):
  """Weighted cross entropy as defined in 
  https://arxiv.org/abs/1902.10903.
  Args:
    logits: Tensor of predictions.
    labels: Tensor of ground truth (same shape as logits).
    lbda: Scalar for positive class weight.
    gamma: Threshold for consensus sampling.
  Returns:
    weighted cross entropy loss
  """
  target = tf.cast(tf.where(tf.greater_equal(labels, gamma),
                            tf.ones_like(labels),
                            labels),
                   tf.float32)
  target_pos = tf.cast(tf.equal(target, 1.), 
                       tf.float32)
  target_neg = tf.cast(tf.equal(target, 0.), 
                       tf.float32)
  pos_ct = tf.reduce_sum(target_pos)
  neg_ct = tf.reduce_sum(target_neg)
  valid = pos_ct + neg_ct
  pos_weight, neg_weight = neg_ct/valid, pos_ct/valid
  weights = lbda*pos_weight*target_pos + neg_weight*target_neg
  xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                 labels=target)
  wce = tf.reduce_mean(weights * xent)
  return wce