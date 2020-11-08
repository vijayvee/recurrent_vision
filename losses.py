"""Custom loss functions."""
import tensorflow.compat.v1 as tf
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
