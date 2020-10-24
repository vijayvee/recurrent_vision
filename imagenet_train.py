"""Trainer for ImageNet classification."""
import os
from absl import app
from absl import flags
import time
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import numpy as np

from recurrent_vision.data_provider import ImageNetDataProvider
from recurrent_vision.models.vgg_config import vgg_config
from recurrent_vision.models.vgg_model import VGG
from recurrent_vision.optimizers import get_optimizer, build_learning_rate

tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.0,
                   "Optimizer learning rate")
flags.DEFINE_float("weight_decay", 1e-4,
                   "weight decay multiplier")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of training epochs")
flags.DEFINE_integer("train_batch_size", 1,
                     "Training minibatch size")
flags.DEFINE_integer("eval_batch_size", 1,
                     "Evaluation minibatch size")
flags.DEFINE_integer("evaluate_every", 1,
                     "Evaluation frequency (every x epochs)")
flags.DEFINE_integer("num_cores", 8,
                     "Number of TPU cores")
flags.DEFINE_integer("iterations_per_loop", 5000,
                     "Number of iterations per TPU loop")
flags.DEFINE_string("experiment_name", "",
                    "Unique experiment identifier")
flags.DEFINE_string("checkpoint", "",
                    "Checkpoint filename")
flags.DEFINE_string("optimizer", "momentum",
                    "Optimizer algorithm (Adam, SGD, etc.)")
flags.DEFINE_boolean("use_tpu", True,
                     "Whether to use TPU for training")
flags.DEFINE_boolean("evaluate", False,
                     "Whether to evaluate during training")
flags.DEFINE_boolean("add_v1net_early", False,
                     "Whether to add v1net after first conv block")
flags.DEFINE_string("tpu_name", "",
                    "Name of TPU to use") 
flags.DEFINE_string("tpu_zone", "europe-west4-a",
                    "TPU zone (europe-west4-a, etc.)")
flags.DEFINE_string("data_dir", "gs://v1net-tpu-bucket/imagenet_data/",
                    "Data directory with ImageNet dataset")


def model_fn(features, labels, mode, params):
  """Build model for boundary detection.
  Args:
    features: (Tensor) of input features, i.e. images.
    labels: (Tensor) of ground truth labels.
    mode: (String) train/eval/predict modes.
    params: (Dict) of model training parameters.
  """
  eval_metrics, train_op, loss = None, None, None
  host_call = None
  training = mode == tf.estimator.ModeKeys.TRAIN
  cfg = vgg_config(add_v1net_early=FLAGS.add_v1net_early)
  vgg = VGG(cfg)
  predictions, _ = vgg.build_model(images=features,
                                   is_training=training,
                                   preprocess=True,
                                   )
  one_hot_labels = tf.one_hot(labels, depth=1000)
  # output predictions
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "predictions": tf.nn.argmax(predictions, axis=1),
        "probabilities": tf.nn.softmax(predictions),
    }
    return tf.estimator.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  loss_fn = tf.nn.softmax_cross_entropy_with_logits
  loss_xent = tf.reduce_mean(loss_fn(logits=predictions,
                                     labels=one_hot_labels,
                                     ))
  # TODO(vveeraba): Test if layer normalization is taken into account below
  loss = loss_xent + FLAGS.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                           if 'batch_normalization' not in v.name])
  
  if training:
    global_step = tf.train.get_global_step()
    steps_per_epoch = params["num_train_steps_per_epoch"]
    learning_rate = build_learning_rate(FLAGS.learning_rate,
                                        global_step,
                                        steps_per_epoch,
                                        decay_factor=0.1,
                                        decay_epochs=30)
    optimizer = get_optimizer(learning_rate,
                              FLAGS.optimizer,
                              FLAGS.use_tpu)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss, global_step)
    train_op = tf.group([train_op, update_ops])
    gs_t = tf.reshape(global_step, [1])
    lr_t = tf.reshape(learning_rate, [1])
    loss_t = tf.reshape(loss, [1])
    predicted_labels = tf.argmax(predictions, 1)
    top_1_acc = tf.metrics.accuracy(predicted_labels, labels)
    top_5_acc = tf.metrics.mean(
                    tf.cast(tf.nn.in_top_k(predictions,
                                           labels,
                                           k=5), 
                            tf.float32))
    top_1_acc = tf.reshape(top_1_acc[0], [1])
    top_5_acc = tf.reshape(top_5_acc[0], [1])

    def host_call_fn(gs, lr, loss, top_1, top_5):
      """Training host call. Creates scalar summaries for training metrics.
      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the
      model to the `metric_fn`, provide as part of the `host_call`. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
      for more information.
      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `host_call`.
      Args:
        gs: `Tensor with shape `[1]` for the global_step
        lr:`Tensor` with shape[1] for learning rate
        loss: `Tensor` with shape `[1]` for the training loss.
        top_1: `Tensor` with shape `[1]` for top-1 accuracy.
        top_5: `Tensor` with shape `[5]` for top-5 accuracy.
      Returns:
        List of summary ops to run on the CPU host.
      """
      gs = gs[0]
      with tf.compat.v2.summary.create_file_writer(params['model_dir'],
						   max_queue=params['iterations_per_loop']).as_default():
        with tf.compat.v2.summary.record_if(True):
          tf.compat.v2.summary.scalar('training/total_loss',loss[0], step=gs)
          tf.compat.v2.summary.scalar('training/learning_rate',lr[0], step=gs)
          tf.compat.v2.summary.scalar('training/top_1_accuracy',top_1[0], step=gs)
          tf.compat.v2.summary.scalar('training/top_5_accuracy',top_5[0], step=gs)
          tf.compat.v2.summary.text('training/training_params',str(params),step=0)
          return tf.summary.all_v2_summary_ops()

    host_call_args = [gs_t, lr_t, loss_t, top_1_acc, top_5_acc]
    host_call = (host_call_fn, host_call_args)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Define evaluation metrics:
    def metric_fn(labels, logits):
      xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,
                                                         logits=logits)
      top_1_accuracy = tf.reduce_sum(tf.nn.in_top_k(logits, labels, k=1))
      top_5_accuracy = tf.reduce_sum(tf.nn.in_top_k(logits, labels, k=5))
      return {
              'xentropy': xentropy,
              'top_1_accuracy': top_1_accuracy,
              'top_5_accuracy': top_5_accuracy,
              }
    eval_metrics = (metric_fn, [labels, predictions])
    
  return tf.estimator.tpu.TPUEstimatorSpec(train_op=train_op,
                                           mode=mode, loss=loss, 
                                           eval_metrics=eval_metrics,
                                           host_call=host_call,
                                           )


def get_input_fn_train(params):
  """Input function for model training."""
  num_examples = 1281167
  def input_fn(params):
    dataset = ImageNetDataProvider(batch_size=params["train_batch_size"],
                                   subset="train",
                                   data_dir=params["data_dir"],
                                   image_size=224,
                                   is_training=True,
                                   )
    images, labels = dataset.images, dataset.labels
    return images, labels
  return num_examples, input_fn

def get_input_fn_validation(params):
  """Input function for model evaluation."""
  num_examples = 50000
  def input_fn(params):
    dataset = ImageNetDataProvider(batch_size=params["eval_batch_size"],
                                   subset="validation",
                                   data_dir=params["data_dir"],
                                   image_size=224,
                                   is_training=False,
                                   )
    images, labels = dataset.images, dataset.labels
    return images, labels
  return num_examples, input_fn


def main(argv):
  del argv  # unused here
  gcs_root = "gs://v1net-tpu-bucket/"
  gcs_path = os.path.join(gcs_root, "imagenet_experiments")
  args = FLAGS.flag_values_dict()
  model_dir = os.path.join(gcs_path, "model_dir_%s" % args["experiment_name"])
  if not tf.gfile.Exists(model_dir):
    tf.gfile.MakeDirs(model_dir)
  args["model_dir"] = model_dir

  rand_seed = np.random.randint(10000)
  tf.set_random_seed(rand_seed)
  args["random_seed"] = rand_seed
  warm_start_settings = None
  num_train_examples, input_fn_train = get_input_fn_train(args)

  args["num_train_examples"] = num_train_examples * args["num_epochs"]
  args["num_train_steps"] = args["num_train_examples"] // args["train_batch_size"]
  num_train_steps = args["num_train_steps"]
  num_train_steps_per_epoch = num_train_steps // args["num_epochs"]
  args["num_train_steps_per_epoch"] = num_train_steps_per_epoch
  
  evaluate_every = int(args["evaluate_every"] * num_train_steps_per_epoch // args["train_batch_size"])
  tf.logging.info("Evaluating every %s steps"%(evaluate_every))
  warm_start_settings = tf.estimator.WarmStartSettings(
                                        ckpt_to_initialize_from=args['checkpoint'],
                                        vars_to_warm_start="^(?!.*side_output|.*Momentum|.*v1net)",
                                        )

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                args["tpu_name"] if args["use_tpu"] else "",
                zone=args["tpu_zone"],
                project="desalab-tpu")
  tpu_config = tf.estimator.tpu.TPUConfig(num_shards=args["num_cores"],
                                          iterations_per_loop=args["iterations_per_loop"])
  save_checkpoints_steps = max(5000, args["iterations_per_loop"])
  config = tf.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                model_dir=model_dir,
                tpu_config=tpu_config,
                save_checkpoints_steps=save_checkpoints_steps,
                keep_checkpoint_max=20)
  classifier = tf.estimator.tpu.TPUEstimator(
                use_tpu=args["use_tpu"],
                model_fn=model_fn,
                config=config,
                params=args,
                warm_start_from=warm_start_settings,
                train_batch_size=args["train_batch_size"],
                eval_batch_size=args["eval_batch_size"],
                )
  start_timestamp = time.time()
  classifier.train(input_fn=input_fn_train, 
                   max_steps=num_train_steps)
  tf.logging.info("Finished training up to step %d. Elapsed seconds %d.",
                  num_train_steps, int(time.time() - start_timestamp))


if __name__=="__main__":
  tf.logging.set_verbosity('INFO')
  app.run(main)
