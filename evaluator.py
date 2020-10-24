"""Evaluation on ImageNet classification."""
from absl import app
from absl import flags

import os
import numpy as np  
import tensorflow.compat.v1 as tf  # pylint: disable=import-error

from recurrent_vision.data_provider import ImageNetDataProvider
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
flags.DEFINE_boolean("preprocess", False,
                     "Whether to add imagenet mean subtraction")


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
  if image.shape[0] == 1:
    # Save only one image
    image = image[0]
  # if image.shape[-1] == 1:
  #   image = image[:,:,0]
  # io.imsave(filename, image)
  cv2.imwrite(filename, image)
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
    predictions, endpoints = self.model.build_model(images, is_training=False,
                                                    preprocess=FLAGS.preprocess)
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
  
  def evaluate_ilsvrc(self):
    """Function to evaluate on ImageNet classification."""
    dataset = ImageNetDataProvider(batch_size=64,
                                   subset="validation",
                                   data_dir=FLAGS.in_dir,
                                   )
    images, labels = dataset.images, dataset.labels
    num_val_examples = dataset.num_examples
    curr_idx = 0
    top_1_acc, top_5_acc = [], []
    if not tf.gfile.Exists(self.out_dir):
      tf.gfile.MakeDirs(self.out_dir)
    outfile = os.path.join(
        self.out_dir, "evaluation_results_%s.txt" % self.model_name)
    predictions  = self.build_model(images)
    with tf.Session() as sess:
      self.model.restore_checkpoint(sess, self.checkpoint_dir)
      top_1_ct = tf.nn.in_top_k(predictions, labels, k=1)
      top_5_ct = tf.nn.in_top_k(predictions, labels, k=5)
      while curr_idx < num_val_examples:
        top_1_np, top_5_np = sess.run([top_1_ct, top_5_ct])
        top_1_acc.append(top_1_np)
        top_5_acc.append(top_5_np)
        curr_idx += len(top_1_np)
        print("Evaluated %s examples \n Top 1: %s Top 5: %s" % (curr_idx,
                                                                round(np.mean(top_1_acc)*100,3),
                                                                round(np.mean(top_5_acc)*100,3)
                                                                )
                                                                )
    print("Top 1 accuracy: %s \nTop 5 accuracy: %s" % (np.mean(top_1_acc),
                                                       np.mean(top_5_acc)))
    with tf.gfile.Open(outfile, "w") as f:
      f.write("Model name: %s \n" % self.model_name)
      metrics = {"top_1_accuracy": np.mean(top_1_acc),
                 "top_5_accuracy": np.mean(top_5_acc),
                 }
      f.write(str(metrics))
    return top_1_acc, top_5_acc


def main(argv):
  del argv  # unused here
  evaluator = Evaluator()
  evaluator.evaluate_ilsvrc()


if __name__=="__main__":
  app.run(main)



