"""Data provider for training models."""
import functools
import multiprocessing
import numpy as np
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import tensorflow_addons as tfa  # pylint: disable=import-error
import tensorflow_datasets as tfds  # pylint: disable=import-error
tf.disable_v2_behavior()

# ImageNet data provider ported from following opensource GitHub repository
# https://github.com/google-research/google-research/blob/master/saccader/data_provider.py

_IMAGE_SIZE_DICT = {
    "imagenet224": 224,
    "imagenet331": 331,
}

# Mean and stddev after normalizing to 0 - 1 range.
_MEAN_RGB_DICT = {
    "imagenet": [0.485, 0.456, 0.406],
}

_STDDEV_RGB_DICT = {
    "imagenet": [0.229, 0.224, 0.225],
}

def augment_images_bsds(image, label):
  """Augment minibatch of images and 
  labels for boundary prediction."""
  image_label_stack = tf.concat([image, label], 
                                 axis=-1)
  rotate_theta = tf.random.uniform((), 0, np.pi/2)
  image_label_stack = tf.image.random_flip_left_right(image_label_stack)
  image_label_stack = tf.image.random_flip_up_down(image_label_stack)
  image_label_stack = tfa.image.rotate(image_label_stack, rotate_theta)
  image_label_unstack = tf.unstack(image_label_stack, axis=-1)
  images_aug = tf.stack(image_label_unstack[:-1], axis=-1)
  labels_aug = tf.expand_dims(image_label_unstack[-1], axis=-1)
  return images_aug, labels_aug

def convert_to_rgb(image):
  """Convert images to RGB from BGR."""
  channels = tf.unstack(image, axis=-1)
  rgb_image = tf.stack([channels[2],
                        channels[1],
                        channels[0]], axis=-1)
  return rgb_image

# =================  Preprocessing Utility Functions. =====================
def _distorted_bounding_box_crop(image,
                                 bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.05, 1.0),
                                 max_attempts=100,
                                 scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.
  See `tf.image.sample_distorted_bounding_box` for more documentation.
  Args:
    image: `Tensor` of image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]` where
      each coordinate is [0, 1) and the coordinates are arranged as `[ymin,
      xmin, ymax, xmax]`. If num_boxes is 0 then use the whole image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped area
      of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image must
      contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope, "distorted_bounding_box_crop", [image, bbox]):
    shape = tf.shape(image)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                          target_height, target_width)

    return image


def _random_crop(image, image_size):
  """Make a random crop of size `image_size`."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=100,
      scope=None)
  return tf.image.resize_bicubic([image], [image_size, image_size])[0]


def _center_crop(image, crop_padding, image_size):
  """Crops to center of image with padding then scales to `image_size`."""
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) * tf.cast(
          tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                        padded_center_crop_size,
                                        padded_center_crop_size)

  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

  return image


def standardize_image(image, dataset):
  """Normalize the image to zero mean and unit variance."""
  moment_shape = [1] * (len(image.shape) - 1) + [3]
  offset = tf.constant(_MEAN_RGB_DICT[dataset], shape=moment_shape)
  image -= offset

  scale = tf.constant(_STDDEV_RGB_DICT[dataset], shape=moment_shape)
  image /= scale
  return image


def preprocess_imagenet_for_train(image, image_size):
  """Preprocesses the given image for evaluation.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    image_size: size of image.
  Returns:
    A preprocessed image `Tensor`.
  """
  image = _random_crop(image, image_size=image_size)
  # image = standardize_image(image, "imagenet")
  image = tf.image.random_flip_left_right(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def preprocess_imagenet_for_eval(image, image_size, crop=True,
                                 standardize=False):
  """Preprocesses the given image for evaluation.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    image_size: size of image.
    crop: If is_training is `False`, determines whether the function should
      extract a central crop of the images (as for standard ImageNet
      evaluation), or rescale the full image without cropping.
    standardize: If `True` (default), standardize to unit variance. Otherwise,
      the returned image is approximately in [0, 1], with some excursions due to
      bicubic resampling.
  Returns:
    A preprocessed image `Tensor`.
  """
  crop_padding = image_size // 10
  image = _center_crop(
      image, crop_padding=crop_padding if crop else 0, image_size=image_size)
  if standardize:
    image = standardize_image(image, "imagenet")
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def preprocess_imagenet(data,
                        is_training,
                        image_size,
                        crop=True):
  """Preprocesses the given image.
  Args:
    data: `Dictionary` with 'image' representing an image of arbitrary size,
      and 'label' representing image class label.
    is_training: `bool` for whether the preprocessing is for training.
    image_size: size of image.
    crop: If is_training is `False`, determines whether the function should
      extract a central crop of the images (as for standard ImageNet
      evaluation), or rescale the full image without cropping.
  Returns:
    A preprocessed image `Tensor`.
    image label.
    mask to track padded vs reral data.
  """
  # Create a mask variable to track the real vs padded data in the last batch.
  mask = 1.
  image = data["image"]

  label = data["label"]
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  if is_training:
    return preprocess_imagenet_for_train(
        image, image_size=image_size), label, mask
  else:
    return preprocess_imagenet_for_eval(
        image, image_size=image_size, crop=crop), label, mask


# ========= ImageNet data provider. ============
class ImageNetDataProvider(object):
  """ImageNet Data Provider.
  Attributes:
    images: (4-D tensor) Images of shape (batch, height, width, channels).
    labels: (1-D tensor) Data labels of size (batch,).
    mask: (1-D boolean tensor) Data mask. Used when data is not repeated to
      indicate the fraction of the batch with true data in the final batch.
    num_classes: (Integer) Number of classes in the dataset.
    num_examples: (Integer) Number of examples in the dataset.
    class_names: (List of Strings) ImageNet id for class labels.
  """

  def __init__(self,
               batch_size,
               subset,
               data_dir,
               image_size=224,
               is_training=False):
    self.image_size = image_size
    dataset_builder = tfds.builder("imagenet2012", data_dir=data_dir)
    dataset_builder.download_and_prepare(download_dir=data_dir)
    if subset == "train":
      dataset = dataset_builder.as_dataset(split=tfds.Split.TRAIN,
                                           shuffle_files=True)
    elif subset == "validation":
      dataset = dataset_builder.as_dataset(split=tfds.Split.VALIDATION)
    else:
      raise ValueError("subset %s is undefined " % subset)
    preprocess_fn = self._preprocess_fn(is_training)
    dataset = dataset.map(preprocess_fn,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    info = dataset_builder.info
    if is_training:
      # 4096 is ~0.625 GB of RAM. Reduce if memory issues encountered.
      dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.repeat(-1 if is_training else 1)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    self.images, self.labels, self.mask = iterator.get_next()
    self.num_classes = info.features["label"].num_classes + 1
    self.class_names = info.features["label"].names
    self.num_examples = info.splits[subset].num_examples

  def _preprocess_fn(self, is_training):
    return functools.partial(preprocess_imagenet, is_training=is_training,
                             image_size=self.image_size)


class BSDSDataProvider:
  """BSDS500 dataset with augmentation pre-built in tfrecords.
  Augmented dataset local copy: /mnt/cube/projects/bsds500/HED-BSDS
  Augmented dataset source: http://vcl.ucsd.edu/hed/HED-BSDS.tar
  """
  def __init__(self,
               batch_size,
               is_training,
               data_dir=None,
               image_size=(400, 400),
               ):
    self.batch_size = batch_size
    self.image_h, self.image_w = image_size
    threads = multiprocessing.cpu_count()
    # load tfrecord files
    if is_training:
      self.training = True
      glob_pattern = "%s/train*" % data_dir
      self.num_examples = 19200
    else:
      self.training=False
      glob_pattern = "%s/validation*" % data_dir
      self.num_examples = 9600
    files = tf.data.Dataset.list_files(glob_pattern, shuffle=is_training)
    # parallel fetching of tfrecords dataset
    dataset = files.apply(tf.data.experimental.parallel_interleave(
                            self.fetch_dataset, cycle_length=threads, 
                            sloppy=True))
    dataset = dataset.map(self.decode_feats, 
                          num_parallel_Calls=tf.data.experimental.AUTOTUNE)
    if is_training:
      # shuffling dataset
      dataset = dataset.shuffle(buffer_size=8 * self.batch_size, 
                                seed=None)
    dataset = dataset.repeat(count=None)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    # use decode function to retrieve images and labels
    #dataset = dataset.apply(
    #            tf.data.experimental.map_and_batch(self.decode_feats,
    #                                               batch_size=self.batch_size,
    #                                               num_parallel_batches=threads,
    #                                               drop_remainder=True))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    self.dataset = dataset

  def fetch_dataset(self, filename):
    """Fetch tf.data.Dataset from tfrecord filename."""
    buffer_size = 8 * 1024 * 1024  # 8 MiB per file
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset

  def decode_feats(self, tfrecord):
    """Decode features written in tfrecords."""
    feat_dict = {
        "height": tf.FixedLenFeature(
            [1], tf.int64),
        "width": tf.FixedLenFeature(
            [1], tf.int64),
        "image_raw": tf.FixedLenFeature(
            [], tf.string),
        "mask_raw": tf.FixedLenFeature(
            [], tf.string),
        "image_path": tf.FixedLenFeature(
            [], tf.string)
        }
    sample = tf.parse_single_example(tfrecord, feat_dict)
    
    # Deserialize data
    img = tf.decode_raw(sample["image_raw"], tf.float32)
    label = tf.decode_raw(sample["mask_raw"], tf.float32)
    height = sample["height"][0]
    width = sample["width"][0]
    img = tf.reshape(img, (height, width, 3))
    label = tf.reshape(label, (height, width, 1))
    img_mask = tf.concat([img, label], axis=-1)
    img_mask = tf.image.resize_with_crop_or_pad(img_mask, 
                                                self.image_h, 
                                                self.image_w)
    img = tf.stack(
              tf.unstack(img_mask, 
                         axis=-1)[:-1],
                         axis=-1)
    label = tf.unstack(img_mask, axis=-1)[-1]
    label = tf.expand_dims(label, axis=2)
    return {"image": img}, {"label": label}
