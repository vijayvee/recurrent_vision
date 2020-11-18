import os

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io  # pylint: disable=import-error
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
from absl import app
from PIL import Image
from tqdm import tqdm

tf.disable_v2_behavior()

import functools
import multiprocessing

from recurrent_vision.models.resnet_v2_config import resnet_v2_config
from recurrent_vision.models.resnet_v2_model import ResNetV2

train_val_file_path = "/mnt/cube/projects/bsds500/HED-BSDS/train_pair.lst"
train_val_files_root = "/mnt/cube/projects/bsds500/HED-BSDS/"

def load_img(img_path):
  img = np.array(Image.open(img_path))
  if img.shape[-1] == 1:
    img = np.repeat(img, 3, axis=-1)
  img = img / 255.
  img = np.expand_dims(img, 0)
  return img

def get_train_val_ids():
  """Get image ids for train and val."""
  train_ids = tf.gfile.ListDirectory("/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/train/")
  val_ids = tf.gfile.ListDirectory("/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/val/")
  train_ids = [i.split(".")[0] for i in train_ids]
  val_ids = [i.split(".")[0] for i in val_ids]
  return train_ids, val_ids

# Helper functions for defining tf types
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_image_annotation_pairs_to_tfrecord(filename_pairs, tfrecords_filename):
  """Writes given image/annotation pairs to the tfrecords file.
  The function reads each image/annotation pair given filenames
  of image and respective annotation and writes it to the tfrecord
  file.
  Parameters
  ----------
  filename_pairs : array of tuples (img_filepath, annotation_filepath)
      Array of tuples of image/annotation filenames
  tfrecords_filename : string
      Tfrecords filename to write the image/annotation pairs
  """
  writer = tf.python_io.TFRecordWriter(tfrecords_filename)
  for path_pair in filename_pairs:
    img_path, annotation_path = path_pair
    cam_path = img_path.replace("jpg", "npy")
    img = np.array(Image.open(os.path.join(train_val_files_root,
                                           img_path)))
    if img.shape[-1] == 1:
      img = np.repeat(img, 3, axis=-1)
    img = img / 255.
    annotation = np.array(Image.open(os.path.join(train_val_files_root,
                                                  annotation_path)))
    cam = np.load(os.path.join(train_val_files_root,
                               cam_path))

    if len(annotation.shape) == 3:
      annotation = annotation[:, :, 0]
    annotation = annotation / 255.
    annotation[annotation>=0.5] = 1
    annotation[annotation!=1] = 0
    assert len(annotation.shape) == 2
    assert tuple(annotation.shape) == tuple(img.shape[:-1])

    img, annotation = np.float32(img), np.float32(annotation)
    cam = np.float32(cam)

    height, width, _ = tuple(img.shape)

    img_raw = img.tostring()
    annotation_raw = annotation.tostring()
    cam_raw = cam.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw),
        'cam_raw': _bytes_feature(cam_raw),
        'image_path': _bytes_feature(img_path.encode('utf-8'))}))

    writer.write(example.SerializeToString())
  writer.close()


def get_image_annotation_pairs(filename):
  """Load image annotation pairs from file."""
  train_ids, val_ids = get_train_val_ids()
  with tf.gfile.Open(filename, "r") as f:
    img_pairs = f.read().strip().split("\n")
  train_pairs, val_pairs = [], []
  for pair in img_pairs:
    img, gt = pair.split(" ")
    img_id = img.split("/")[-1].split(".")[0]
    if img_id in train_ids:
      train_pairs.append((img, gt))
    else:
      val_pairs.append((img, gt))
  return train_pairs, val_pairs


def write_tfrecords_train(filename, num_train_shards=100):
  base_tfr_path = "/mnt/cube/projects/bsds500/HED-BSDS/tfrecords"
  train_tfr_filename = "cam_train"
  train_pairs, _ = get_image_annotation_pairs(filename)
  np.random.shuffle(train_pairs)
  shard_step = len(train_pairs) // num_train_shards
  for shard_idx in np.arange(0, len(train_pairs), shard_step):
    curr_batch = train_pairs[shard_idx: shard_idx+shard_step]
    curr_idx = shard_idx // shard_step + 1
    curr_tfr_filename = "%s-%s-of-%s" % (train_tfr_filename,
                                         curr_idx,
                                         num_train_shards)
    curr_tfr_filename = os.path.join(base_tfr_path,
                                     curr_tfr_filename)
    print("Writing %s .." % curr_tfr_filename)
    write_image_annotation_pairs_to_tfrecord(curr_batch,
                                             curr_tfr_filename)

def write_tfrecords_validation(filename, num_val_shards=100):
  base_tfr_path = "/mnt/cube/projects/bsds500/HED-BSDS/tfrecords"
  val_tfr_filename = "cam_validation"
  _, val_pairs = get_image_annotation_pairs(filename)
  np.random.shuffle(val_pairs)
  shard_step = len(val_pairs) // num_val_shards
  for shard_idx in np.arange(0, len(val_pairs), shard_step):
    curr_batch = val_pairs[shard_idx: shard_idx+shard_step]
    curr_idx = shard_idx // shard_step + 1
    curr_tfr_filename = "%s-%s-of-%s" % (val_tfr_filename,
                                         curr_idx,
                                         num_val_shards)
    curr_tfr_filename = os.path.join(base_tfr_path,
                                     curr_tfr_filename)
    print("Writing %s .." % curr_tfr_filename)
    write_image_annotation_pairs_to_tfrecord(curr_batch,
                                             curr_tfr_filename)


def main(argv):
  del argv  # unused
  write_tfrecords_train(train_val_file_path)
  write_tfrecords_validation(train_val_file_path)


if __name__=="__main__":
  app.run(main)