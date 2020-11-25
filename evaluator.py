"""Evaluation on ImageNet classification."""
import os

import cv2  # pylint: disable=import-error
import numpy as np
import skimage.io as io  # pylint: disable=import-error
from scipy.io import savemat
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
from tqdm import tqdm
from absl import app, flags
from PIL import Image

from recurrent_vision.data_provider import ImageNetDataProvider
from recurrent_vision.models.vgg16_hed_config import vgg_16_hed_config
from recurrent_vision.models.vgg_config import vgg_config
from recurrent_vision.models.vgg_model import VGG
from recurrent_vision.models.vgg_v1net_config import vgg_v1net_config

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
flags.DEFINE_string("gcs_dir", "",
                    "Directory where checkpoints are "
                    "stored for a suite of models on GCS")
flags.DEFINE_string("model_dir", "",
                    "Single model directory to evaluate")
flags.DEFINE_boolean("preprocess", True,
                     "Whether to add imagenet mean subtraction")
flags.DEFINE_boolean("add_cam", False,
                     "Whether to add CAM")
flags.DEFINE_boolean("add_v1net_early", False,
                     "Whether to add V1Net")
flags.DEFINE_boolean("add_v1net", False,
                     "Whether to add V1Net throughout")
flags.DEFINE_integer("v1_timesteps", 4,
                     "Number of V1Net timesteps")

def load_image(image_path):
  """Load images from disk."""
  img = np.array(Image.open(image_path))
  transpose = False
  if img.max() > 1.:
    img = img / 255.
  if img.shape[-1] == 1:
    img = np.repeat(img, 3, axis=-1)
  if img.shape[0] == 481:
    img = np.transpose(img, (1, 0, 2))
    transpose = True
  img = np.expand_dims(img, axis=0)
  return img, transpose


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
    # Saves only one image
    image = image[0]
  if image.shape[-1] == 1:
    image = image[:,:,0]
  image = np.uint8(image*255.)
  io.imsave(filename, image)
  return filename

def save_mat(image, prefix=None,
             path=None, curr_idx=None):
  """Write images to disk."""
  if curr_idx:
    filename = "%s_%04d.mat" % (prefix.split('.')[0], 
                                curr_idx)
  else:
    filename = "%s.mat" % prefix.split('.')[0]
  filename = os.path.join(path, filename)
  if image.shape[0] == 1:
    # Saves only one image
    image = image[0]
  if image.shape[-1] == 1:
    image = image[:,:,0]
  mat_dict = {"predictions": image}
  savemat(filename, mat_dict)
  return filename


class Evaluator:
  """Evaluate models on ImageNet classification."""
  def __init__(self):
    self.model_name = FLAGS.model_name
    self.in_dir = FLAGS.in_dir
    self.out_dir = FLAGS.out_dir
    self.checkpoint_dir = FLAGS.checkpoint_dir
    self.gcs_dir = FLAGS.gcs_dir

  def build_model(self, images, cams):
    """Build imagenet classification model."""
    model_config = None
    if self.model_name.startswith("vgg_16_hed"):
      model_config = vgg_16_hed_config(add_v1net_early=FLAGS.add_v1net_early,
                                       add_v1net=FLAGS.add_v1net,
                                       cam_net=FLAGS.add_cam)
    elif self.model_name.startswith("vgg_19"):
      model_config = vgg_config(vgg_depth=19)
    elif self.model_name.startswith("vgg_16"):
      model_config = vgg_config()
    self.model = VGG(model_config)
    predictions, _ = self.model.build_model(images, cams=cams,
                                            is_training=False,
                                            preprocess=FLAGS.preprocess)
    return predictions

  def evaluate(self, in_dir, out_dir, checkpoint=None):
    """Function to write boundary predictions."""
    if not checkpoint:
      checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
    img_fns = tf.gfile.ListDirectory(in_dir)
    img_fns = [i for i in img_fns if "jpg" in i]
    cam_fns = [i.replace("jpg", "npy") for i in img_fns]
    if not tf.gfile.Exists(out_dir):
      tf.gfile.MakeDirs(out_dir)
    with tf.Graph().as_default(), tf.Session() as sess:
      # Build and restore model
      images = tf.placeholder(tf.float32, [1, 321, 481, 3])
      cams = tf.placeholder(tf.float32, [1, 321, 481, 1])
      predictions = tf.nn.sigmoid(self.build_model(images, cams))
      self.model.restore_checkpoint(sess, checkpoint)
      # Generate predictions
      for img_fn, cam_fn in tqdm(zip(img_fns, cam_fns)):
        img, transpose = load_image(os.path.join(in_dir, img_fn))
        print(img.max())
        cam = np.expand_dims(np.load(os.path.join(in_dir, cam_fn)), axis=(0, 3))
        if transpose:
          cam = np.transpose(cam, (0, 2, 1, 3))
        model_pred = sess.run(predictions, 
                              feed_dict={images: img,
                                         cams: cam})
        if transpose:
          model_pred = np.transpose(model_pred, 
                                    (0, 2, 1, 3))
        save_mat(model_pred, prefix=img_fn, 
                   path=out_dir)

  def evaluate_recursive(self):
    """Function to write boundary 
       predictions for all checkpoints."""
    in_dir = self.in_dir
    model_dirs = tf.gfile.ListDirectory(self.gcs_dir)
    model_dirs = [i for i in model_dirs 
                  if i.startswith("model_dir")]
    if FLAGS.model_dir is not "":
      model_dirs = [FLAGS.model_dir]
    tf.logging.info("Found %s model directories" % len(model_dirs))
    for m_d in model_dirs:
      print("Evaluating %s" % m_d)
      out_dir = os.path.join(self.out_dir, m_d)
      if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)
      checkpoints = tf.gfile.ListDirectory(
                          os.path.join(self.gcs_dir, m_d))
      checkpoints = [i for i in checkpoints
                      if ".ckpt" in i and ".meta" in i]
      for checkpoint in checkpoints:
        # Don't evaluate checkpoint-0 (rand init)
        if "model.ckpt-0" in checkpoint:
          continue
        ckpt_idx = int(checkpoint.split('ckpt-')[1].split('.')[0])
        if ckpt_idx % 10000:
          continue
        checkpoint = checkpoint.split(".meta")[0]
        sub_out_dir = os.path.join(out_dir, checkpoint)
        if not tf.gfile.Exists(sub_out_dir):
          tf.gfile.MakeDirs(sub_out_dir)
        checkpoint = os.path.join(self.gcs_dir, m_d, checkpoint)
        self.evaluate(in_dir, sub_out_dir, checkpoint)
  
  def evaluate_ilsvrc(self):
    """Function to evaluate on ImageNet classification."""
    dataset = ImageNetDataProvider(batch_size=64,
                                   subset="validation",
                                   data_dir=FLAGS.in_dir,
                                   is_training=False,
                                   )
    images, labels = dataset.images, dataset.labels
    num_val_examples = dataset.num_examples
    curr_idx = 0
    top_1_acc, top_5_acc = [], []
    if not tf.gfile.Exists(self.out_dir):
      tf.gfile.MakeDirs(self.out_dir)
    outfile = os.path.join(
        self.out_dir, "evaluation_results_%s.txt" % self.model_name)
    predictions  = self.build_model(images, cams=None)
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
  evaluator.evaluate_recursive()


if __name__=="__main__":
  app.run(main)



