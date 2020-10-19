"""Unit tests for v1net_model.py"""
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class V1NetCNNTest(tf.test.TestCase):
  """Class to unit test V1Net model."""

  def test_endpoint_shape(self):
    # TODO(vveeraba): Test the shape of output from V1NetCNN
    pass
