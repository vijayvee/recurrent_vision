"""Trainer for BSDS boundary detection."""
import os
from absl import app
from absl import flags
import time
import tensorflow.compat.v1 as tf  # pylint: disable=import-error
import numpy as np

from recurrent_vision.data_provider import BSDSDataProvider
from recurrent_vision.models.vgg16_hed_config import vgg_16_hed_config
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
flags.DEFINE_integer("num_cores", 8,
                     "Number of TPU cores")
flags.DEFINE_integer("iterations_per_loop", 500,
                     "Number of iterations per TPU loop")
flags.DEFINE_string("experiment_name", "",
                    "Unique experiment identifier")
flags.DEFINE_string("checkpoint", "",
                    "Checkpoint filename")
flags.DEFINE_string("optimizer", "",
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
flags.DEFINE_string("data_dir", "",
                    "Data directory with BSDS500 tfrecords")

# TODO(vveeraba): add learning rate decay
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
  eval_metrics, train_op, loss = None, None, None
  host_call = None
  training = mode == tf.estimator.ModeKeys.TRAIN
  cfg = vgg_16_hed_config(add_v1net_early=FLAGS.add_v1net_early)
  vgg = VGG(cfg)
  predictions, endpoints = vgg.build_model(images=features["image"],
                                           is_training=training,
                                           preprocess=False)
  # TODO(vveeraba): Add vgg restore checkpoint
  # Tile ground truth for 5 side outputs
  side_predictions = endpoints["side_outputs_fullres"]
  side_labels = tf.tile(labels["label"], [5, 1, 1, 1])

  # output predictions
  if mode == tf.estimator.ModeKeys.PREDICT:
    sigmoid = tf.nn.sigmoid(predictions)
    predictions = {
        "predictions": predictions,
        "boundary_pred_map": sigmoid,
    }
    return tf.estimator.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  # TODO(vveeraba): Change positive class weight below
  pos_weight = 1.1
  loss_fn = tf.nn.weighted_cross_entropy_with_logits
  xent = tf.nn.sigmoid_cross_entropy_with_logits
  loss_fuse = tf.reduce_mean(xent(logits=predictions,
                                  labels=labels["label"],
                                  ))
  loss_side = tf.reduce_mean(loss_fn(logits=side_predictions,
                                     labels=side_labels,
                                     pos_weight=pos_weight),
                                     )
  loss = loss_side + loss_fuse + FLAGS.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                           if 'normalization' not in v.name])
  
  if training:
    global_step = tf.train.get_global_step()
    steps_per_epoch = params["num_train_steps_per_epoch"]
    learning_rate = build_learning_rate(FLAGS.learning_rate,
                                        global_step,
                                        steps_per_epoch,
                                        decay_factor=0.1,
                                        decay_epochs=25)
    fast_start = min(FLAGS.learning_rate*100, 1e-4)
    fast_learning_rate = build_learning_rate(fast_start,
                                             global_step,
                                             steps_per_epoch,
                                             decay_factor=0.1,
                                             decay_epochs=10)

    optimizer = get_optimizer(learning_rate,
                              FLAGS.optimizer,
                              FLAGS.use_tpu)
    slow_vars = [var for var in vgg.model_vars 
                    if "v1net" not in var.name]
    fast_vars = list(set(vgg.model_vars).difference(set(slow_vars)))
    fast_optimizer = get_optimizer(fast_learning_rate,
                                   FLAGS.optimizer,
                                   FLAGS.use_tpu)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    train_op = optimizer.minimize(loss, global_step, var_list=slow_vars)
    fast_train_op = fast_optimizer.minimize(loss, global_step, var_list=fast_vars)

    train_op = tf.group([train_op, update_ops, fast_train_op])

    gs_t = tf.reshape(global_step, [1])
    lr_t = tf.reshape(learning_rate, [1])
    fast_lr_t = tf.reshape(fast_learning_rate, [1])
    loss_t = tf.reshape(loss, [1])
    loss_side_t = tf.reshape(loss_side, [1])
    loss_fuse_t = tf.reshape(loss_fuse, [1])
    labels_t = labels["label"]
    preds_t = tf.nn.sigmoid(predictions)

    def host_call_fn(gs, lr, fast_lr, loss, loss_side, loss_fuse, 
	             lbls, preds):
      """Training host call. Creates scalar summaries for training metrics.
      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the
      model to the `metric_fn`, provide as part of the `host_call`. See
      https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
      for more information.
      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `host_call`.
      Args:
	gs: `Tensor with shape `[batch]` for the global_step
	loss: `Tensor` with shape `[batch]` for the training loss.
	loss_side: `Tensor` with shape `[batch]` for the training side loss.
	loss_fuse: `Tensor` with shape `[batch]` for the training fused loss.
	img: `Tensor` of input images.
      Returns:
	List of summary ops to run on the CPU host.
      """
      gs = gs[0]
      with tf.compat.v2.summary.create_file_writer(params['model_dir'],
						   max_queue=params['iterations_per_loop']
                                                   ).as_default():
        with tf.compat.v2.summary.record_if(True):
          tf.compat.v2.summary.scalar('training/total_loss',loss[0], step=gs)
          tf.compat.v2.summary.scalar('training/side_loss',loss_side[0], step=gs)
          tf.compat.v2.summary.scalar('training/fuse_loss',loss_fuse[0], step=gs)
          tf.compat.v2.summary.scalar('training/learning_rate',lr[0], step=gs)
          tf.compat.v2.summary.scalar('training/fast_learning_rate',fast_lr[0], step=gs)
          tf.compat.v2.summary.image('training/predictions',1-preds,step=gs)
          tf.compat.v2.summary.image('training/labels',lbls,step=gs)
          return tf.summary.all_v2_summary_ops()

    host_call_args = [gs_t, lr_t, fast_lr_t, loss_t, loss_side_t, 
                      loss_fuse_t, labels_t, preds_t]
    host_call = (host_call_fn, host_call_args)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Define evaluation metrics:
    def metric_fn(labels, logits):
      xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels['label'],
                                                         logits=logits)
      return {
              'xentropy': xentropy,
              }
    label_tensor = labels['label']
    eval_metrics = (metric_fn, [label_tensor, predictions])
    
  return tf.estimator.tpu.TPUEstimatorSpec(train_op=train_op,
                                           mode=mode, loss=loss, 
                                           eval_metrics=eval_metrics,
                                           host_call=host_call,
                                           )


def get_input_fn_train(params):
  """Input function for model training."""
  num_examples = 19200
  def input_fn(params):
    dataset = BSDSDataProvider(params["train_batch_size"],
                               is_training=True,
                               data_dir=params["data_dir"])
    return dataset.dataset
  return num_examples, input_fn

def get_input_fn_validation(params):
  """Input function for model evaluation."""
  num_examples = 9600
  def input_fn(params):
    dataset = BSDSDataProvider(params["eval_batch_size"],
                             is_training=False,
                             data_dir=params["data_dir"])
    return dataset.dataset
  return num_examples, input_fn

def get_scaffold_fn():
  """Scaffold function for initialization."""
  def scaffold_fn():
    latest_checkpoint = FLAGS.checkpoint
    model_vars = [var.name[:-2] for var in list(tf.global_variables())]
    # Get checkpoint vars
    ckpt_vars = list(tf.train.list_variables(latest_checkpoint))
    ckpt_vars = [var for var, var_shape in ckpt_vars]
    # Restore vars = checkpoint_vars.intersection(model_vars)
    restore_vars = list(set(ckpt_vars).intersection(set(model_vars)))
    restore_vars = {var.name[:-2]: var for var in tf.global_variables()
                    if var.name[:-2] in restore_vars}
    tf.train.init_from_checkpoint(latest_checkpoint, restore_vars)
    return tf.train.Scaffold()
  return scaffold_fn


def main(argv):
  del argv  # unused here
  gcs_root = "gs://v1net-tpu-bucket/"
  gcs_path = "gs://v1net-tpu-bucket/bsds_experiments/"
  args = FLAGS.flag_values_dict()
  model_dir = os.path.join(gcs_path, "model_dir_%s" % args["experiment_name"])
  if not tf.gfile.Exists(model_dir):
    tf.gfile.MakeDirs(model_dir)
  args["model_dir"] = model_dir

  rand_seed = np.random.randint(10000)
  tf.set_random_seed(rand_seed)
  args["random_seed"] = rand_seed
  warm_start_settings = None
  args["data_dir"] = os.path.join(gcs_root, args["data_dir"])

  num_train_examples, input_fn_train = get_input_fn_train(args)

  args["num_train_examples"] = num_train_examples * args["num_epochs"]
  args["num_train_steps"] = args["num_train_examples"] // args["train_batch_size"]
  num_train_steps = args["num_train_steps"]
  num_train_steps_per_epoch = num_train_steps // args["num_epochs"]
  args["num_train_steps_per_epoch"] = num_train_steps_per_epoch
  
  warm_start_settings = tf.estimator.WarmStartSettings(
                                        ckpt_to_initialize_from=args['checkpoint'],
                                        vars_to_warm_start=["^(?!.*side_output|.*v1net|.*Momentum|global_step|beta*|gamma*|.*Adam)"],
                                        )

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                args["tpu_name"] if args["use_tpu"] else "",
                zone=args["tpu_zone"],
                project="desalab-tpu")
  tpu_config = tf.estimator.tpu.TPUConfig(num_shards=args["num_cores"],
                                          iterations_per_loop=args["iterations_per_loop"])
  config = tf.estimator.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                model_dir=model_dir,
                tpu_config=tpu_config,
                save_checkpoints_steps=args["iterations_per_loop"],
                keep_checkpoint_max=20)

  classifier = tf.estimator.tpu.TPUEstimator(
                use_tpu=args["use_tpu"],
                model_fn=model_fn,
                config=config,
                params=args,
                warm_start_from=warm_start_settings,
                train_batch_size=args["train_batch_size"],
                )

  start_timestamp = time.time()
  classifier.train(
        input_fn=input_fn_train, max_steps=num_train_steps)
  elapsed_time = int(time.time() - start_timestamp)
  tf.logging.info("Finished training up to step %d. Elapsed seconds %d.",
                          num_train_steps, elapsed_time)

if __name__=="__main__":
  tf.logging.set_verbosity('INFO')
  app.run(main)


