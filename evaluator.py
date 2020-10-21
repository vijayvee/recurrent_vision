"""Evaluation on ImageNet classification."""
from absl import app
from absl import flags

import cv2  # pylint: disable=import-error
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error

from tqdm import tqdm
from PIL import Image
from recurrent_vision.models.vgg16_hed_config import vgg_16_hed_config
from recurrent_vision.models.vgg_v1net_config import vgg_v1net_config
from recurrent_vision.models.vgg_config import vgg_config
from recurrent_vision.models.vgg_model import VGG
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("model_name", "vgg_16_hed",
                    "Name of model to build")
flags.DEFINE_string("in_dir", "",
                    "Directory where test images stored")
flags.DEFINE_string("out_dir", "",
                    "Directory where predictions to be written")
flags.DEFINE_string("checkpoint_dir", "",
                    "Directory where checkpoints are stored")


def load_image(image_path):
  """Load images from disk."""
  img = np.array(Image.open(image_path))
  if img.max() > 1.:
    img = img / 255.
  if img.shape[-1] == 1:
    img = np.repeat(img, 3, axis=-1)
  if img.shape[0] == 481:
    img = np.transpose(img, (1, 0, 2))
  img = np.expand_dims(img, axis=0)
  return img


def save_image(image, prefix=None,
               path=None, curr_idx=None):
  """Write images to disk."""
  if curr_idx:
    filename = "%s_%04d.png" % (prefix.split('.')[0], 
                                curr_idx)
  else:
    filename = "%s.png" % prefix.split('.')[0]
  filename = os.path.join(path, filename)
  if len(image.shape) == 4:
    # Save only one image
    image = image[0]
  cv2.imwrite(filename, image*255)
  # io.imsave(filename, image)
  return filename


class Evaluator:
  """Evaluate models on ImageNet classification."""
  def __init__(self):
    self.model_name = FLAGS.model_name
    self.in_dir = FLAGS.in_dir
    self.out_dir = FLAGS.out_dir
    self.checkpoint_dir = FLAGS.checkpoint_dir

  def build_model(self, images):
    """Build imagenet classification model."""
    model_config = None
    if self.model_name.startswith("vgg_16_hed"):
      model_config = vgg_16_hed_config()
    elif self.model_name.startswith("vgg_19"):
      model_config = vgg_config(vgg_depth=19)
    elif self.model_name.startswith("vgg_16"):
      model_config = vgg_config()
    self.model = VGG(model_config)
    _, endpoints = self.model.build_model(images, is_training=False)
    predictions = endpoints["test_outputs"]
    return predictions

  def evaluate(self):
    """Function to write boundary predictions."""
    in_dir = self.in_dir
    out_dir = self.out_dir
    img_fns = tf.gfile.ListDirectory(in_dir)
    img_fns = [i for i in img_fns if "jpg" in i]
    if not tf.gfile.Exists(out_dir):
      tf.gfile.MakeDirs(out_dir)
    with tf.Session() as sess:
      # Build and restore model
      images = tf.placeholder(tf.float32, [1, 321, 481, 3])
      predictions = tf.nn.sigmoid(self.build_model(images)) 
      latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
      self.model.restore_checkpoint(sess, latest_checkpoint)

      # Generate predictions
      for img_fn in tqdm(img_fns):
        img = load_image(os.path.join(in_dir, img_fn))
        model_pred = sess.run(predictions, 
                              feed_dict={images: img})
        save_image(model_pred, prefix=img_fn, 
                   path=out_dir)
  

def main(argv):
  del argv  # unused here
  evaluator = Evaluator()
  evaluator.evaluate()


if __name__=="__main__":
  app.run(main)



