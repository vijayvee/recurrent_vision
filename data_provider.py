"""Data provider for training models."""
import multiprocessing
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error

tf.disable_v2_behavior()


class BSDSDataProvider:
  """BSDS500 dataset."""
  def __init__(self,
               batch_size,
               is_training,
               data_dir=None,
               ):
    self.batch_size = batch_size
    threads = multiprocessing.cpu_count()
    # load tfrecord files
    if is_training:
      glob_pattern = "%s/train*.tfrecord" % data_dir
      self.num_examples = 300
    else:
      glob_pattern = "%s/val*.tfrecord" % data_dir
      self.num_examples = 100
    files = tf.data.Dataset.list_files(glob_pattern, shuffle=is_training)
    # parallel fetching of tfrecords dataset
    dataset = files.apply(tf.data.experimental.parallel_interleave(
                            self.fetch_dataset, cycle_length=threads, sloppy=True))
    # shuffling dataset
    dataset = dataset.shuffle(buffer_size=8 * self.batch_size, seed=None)
    dataset = dataset.repeat(count=None)
    # use decode function to retrieve images and labels
    dataset = dataset.map(map_func=lambda example: self.decode_feats(example,
                                                                     is_training),
                          num_parallel_calls=threads)
    # batch the examples
    dataset = dataset.batch(batch_size=self.batch_size)
    self.images, self.labels = dataset.make_one_shot_iterator().get_next()

  def fetch_dataset(self, filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

  def decode_feats(self, tfrecord, training):
    """Decode features written in tfrecords."""
    split = "train"
    if not training:
        split = "val"
    feat_dict = {
        "%s/shape" %(split): tf.FixedLenFeature(
            [2], tf.int64),
        "%s/image" %(split): tf.FixedLenFeature(
            [], tf.string),
        "%s/label" %(split): tf.FixedLenFeature(
            [], tf.string)
        }
    sample = tf.parse_single_example(tfrecord, feat_dict)
    img = tf.decode_raw(sample["%s/image" %(split)] ,tf.float64)
    img = tf.cast(img, tf.float32)
    label = tf.decode_raw(sample["%s/label" %(split)] ,tf.float32)
    img = tf.reshape(img, [321, 481, 3])
    label = tf.reshape(label, [321, 481, 1])
    return {"image": img}, {"label": label}
  
  def input_fn(self, params):
    """Return tensors for input and labels."""
    return self.images, self.labels
