"""Trainer for BSDS boundary detection."""
import os
from absl import app
from absl import flags
from datetime import time
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import numpy as np

from recurrent_vision.data_provider import BSDSDataProvider
from recurrent_vision.models.vgg16_hed_config import vgg_16_hed_config
from recurrent_vision.models.vgg_model import VGG
from recurrent_vision.optimizers import get_optimizer

FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.0,
                   "Optimizer learning rate")
flags.DEFINE_float("weight_decay", 1e-4,
                   "weight decay multiplier")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of training epochs")
flags.DEFINE_integer("batch_size", 1,
                     "Mini batch size")
flags.DEFINE_integer("evaluate_every", 1,
                     "Evaluation frequency (every x epochs)")
flags.DEFINE_string("experiment_name", "",
                    "Unique experiment identifier")
flags.DEFINE_string("optimizer", "",
                    "Optimizer algorithm (Adam, SGD, etc.)")
flags.DEFINE_boolean("use_tpu", True,
                     "Whether to use TPU for training")
flags.DEFINE_string("tpu_name", "",
                    "Name of TPU to use")

# TODO(vveeraba): add learning rate decay
# TODO(vveeraba): add checkpoint restoring
# TODO(vveeraba): add dropout
# TODO(vveeraba): check efficientnet main.py and implement features

def model_fn(features, labels, mode, params):
  """Build model for boundary detection.
  Args:
    features: (Tensor) of input features, i.e. images.
    labels: (Tensor) of ground truth labels.
    mode: (String) train/eval/predict modes.
    params: (Dict) of model training parameters.
  """
  training = mode == tf.estimator.ModeKeys.TRAIN
  cfg = vgg_16_hed_config()
  vgg = VGG(cfg)
  predictions, endpoints = vgg.build_model(images=features["image"],
                                           is_training=training)
  # TODO(vveeraba): Add vgg restore checkpoint
  # Tile ground truth for 5 side outputs
  side_predictions = endpoints["side_outputs_fullres"]
  side_labels = tf.tile(labels, [5, 1, 1, 1])

  # output predictions
  if mode == tf.estimator.ModeKeys.PREDICT:
    # TODO(vveerabadran): implement aggregate predictions (from fusion)
    predictions = tf.reduce_mean(tf.stack([predictions,
                                           side_predictions],
                                           axis=0),
                                           axis=0,)
    sigmoid = tf.nn.sigmoid(predictions)
    predictions = {
        "predictions": predictions,
        "boundary_pred_map": sigmoid,
    }
    return tf.estimator.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  pos_weight = 0.9
  loss_fn = tf.nn.weighted_cross_entropy_with_logits
  loss_fuse = loss_fn(logits=predictions,
                      labels=labels["label"],
                      pos_weight=pos_weight,
                      )
  loss_side = loss_fn(logits=side_predictions,
                      labels=side_labels,
                      pos_weight=pos_weight
                      )
  loss = loss_side + loss_fuse
  
  if training:
    global_step = tf.train.get_global_step()
    optimizer = get_optimizer(FLAGS.learning_rate,
                              FLAGS.optimizer,
                              FLAGS.use_tpu)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss, global_step)
    train_op = tf.group([train_op, update_ops])

  if mode == tf.estimator.ModeKeys.EVAL:
    # Define evaluation metrics:
    def metric_fn(labels, logits):
      xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                         logits=logits)
      return {
              'xentropy': xentropy,
              }
    label_tensor = labels['label']
    eval_metrics = (metric_fn, [label_tensor, predictions])
    
  return tf.estimator.tpu.TPUEstimatorSpec(train_op=train_op,
                                           mode=mode, loss=loss, 
                                           eval_metrics=eval_metrics,
                                           )

def get_input_fn_train(params):
  """Input function for data serving during model training."""
  dataset = BSDSDataProvider(params["batch_size"],
                             is_training=True,
                             data_dir=params["data_dir"])
  return dataset, dataset.input_fn

def get_input_fn_validation(params):
  """Input function for data serving during model evaluation."""
  dataset = BSDSDataProvider(params["batch_size"],
                             is_training=False,
                             data_dir=params["data_dir"])
  return dataset, dataset.input_fn

def main():
  gcs_path = "gs://v1net-tpu-bucket/bsds_experiments/"
  args = FLAGS.flag_values_dict()
  model_dir = os.path.join(gcs_path, "model_dir_%s" % args["expt_name"])
  args["model_dir"] = model_dir

  rand_seed = np.random.randint(10000)
  tf.set_random_seed(rand_seed)
  args["random_seed"] = rand_seed
  warm_start_settings = None

  dataset_train, input_fn_train = get_input_fn_train(args)
  dataset_val, input_fn_val = get_input_fn_validation(args)
  args["num_train_examples"] = dataset_train.num_examples * args["num_epochs"]
  args["num_train_steps"] = args["num_train_examples"] // args["batch_size"]
  num_train_steps = args["num_train_steps"]
  num_train_steps_per_epoch = num_train_steps // args["num_epochs"]
  
  args["num_eval_examples"] = dataset_val.num_examples
  args["num_eval_steps"] = args["num_eval_examples"] // args["batch_size"]
  num_eval_steps = args["num_eval_steps"]

  eval_every = int(args["eval_every"] * num_train_steps_per_epoch // args["batch_size"])
  tf.logging.info("Evaluating every %s steps"%(eval_every))

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                args["tpu_name"] if args["use_tpu"] else "",
                zone="europe-west4-a",
                project="desalab-tpu")
  config = tf.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                model_dir=model_dir,
                tpu_config=tf.estimator.tpu.TPUConfig(
                                          # Since we use vx-8, i.e, 8 cores of vx tpu
                                          num_shards=8,
                                          iterations_per_loop=100))

  classifier = tf.estimator.tpu.TPUEstimator(
                use_tpu=args["use_tpu"],
                model_fn=model_fn,
                config=config,
                params=args,
                warm_start_from=warm_start_settings,
                train_batch_size=args["batch_size"],
                eval_batch_size=args["batch_size"],
                )

  try:
    current_step = tf.train.load_variable(model_dir,
                                          tf.GraphKeys.GLOBAL_STEP)
  except (TypeError, ValueError, tf.errors.NotFoundError):
    current_step = 0
  start_timestamp = time.time()
  while current_step < num_train_steps:
    # Train until next evaluation step
    next_checkpoint = min(current_step + eval_every,
                          num_train_steps)
    classifier.train(
        input_fn=input_fn_train, max_steps=next_checkpoint)
    current_step = next_checkpoint

    tf.logging.info("Finished training up to step %d. Elapsed seconds %d.",
                    next_checkpoint, int(time.time() - start_timestamp))

    tf.logging.info("Starting to evaluate.")
    eval_results = classifier.evaluate(
                        input_fn=input_fn_val,
                        steps=num_eval_steps)
    tf.logging.info("Eval results at step %d: %s",
                    next_checkpoint, eval_results)

  elapsed_time = int(time.time() - start_timestamp)
  tf.logging.info("Finished training up to step %d. Elapsed seconds %d.",
                          num_train_steps, elapsed_time)

if __name__=="__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main())


