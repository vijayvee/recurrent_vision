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
from recurrent_vision.losses import weighted_ce, weighted_ce_bdcn


tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 0.0,
                   "Optimizer learning rate")
flags.DEFINE_float("weight_decay", 1e-4,
                   "weight decay multiplier")
flags.DEFINE_float("evaluate_every", 0.01,
                   "Evaluation frequency in epochs")
flags.DEFINE_float("label_gamma", 0.4,
                   "Gamma for label consensus sampling")
flags.DEFINE_float("label_lbda", 1.1,
                   "Positive weight for wce")
flags.DEFINE_integer("num_epochs", 100,
                     "Number of training epochs")
flags.DEFINE_integer("train_batch_size", 1,
                     "Training minibatch size")
flags.DEFINE_integer("eval_batch_size", 1,
                     "Evaluation minibatch size")
flags.DEFINE_integer("num_cores", 8,
                     "Number of TPU cores")
flags.DEFINE_integer("iterations_per_loop", 5000,
                     "Number of iterations per TPU loop")
flags.DEFINE_integer("iterations_per_checkpoint", 10000,
                     "Number of iterations per checkpoint")
flags.DEFINE_integer("image_size", 500,
                     "Input image size")
flags.DEFINE_integer("v1_timesteps", 4,
                     "Number of V1Net timesteps")
flags.DEFINE_integer("max_steps", 60000,
                     "Maximum number of steps before stopping")
flags.DEFINE_string("experiment_name", "",
                    "Unique experiment identifier")
flags.DEFINE_string("checkpoint", "",
                    "Checkpoint filename")
flags.DEFINE_string("optimizer", "",
                    "Optimizer algorithm (Adam, SGD, etc.)")
flags.DEFINE_string("base_dir", "bsds_experiments",
                    "Base directory to store experiments on GCS")
flags.DEFINE_boolean("use_tpu", True,
                     "Whether to use TPU for training")
flags.DEFINE_boolean("add_cam", False,
                     "Whether to add CAM input")
flags.DEFINE_boolean("evaluate", False,
                     "Whether to evaluate during training")
flags.DEFINE_boolean("add_v1net_early", False,
                     "Whether to add v1net after first conv block")
flags.DEFINE_boolean("add_v1net", False,
                     "Whether to add v1net throughout")
flags.DEFINE_boolean("train_and_eval", False,
                     "Whether to evaluate between training epochs")
flags.DEFINE_boolean("preprocess", False,
                     "Whether to add preprocessing")
flags.DEFINE_string("tpu_name", "",
                    "Name of TPU to use")
flags.DEFINE_string("tpu_zone", "europe-west4-a",
                    "TPU zone (europe-west4-a, etc.)")
flags.DEFINE_string("data_dir", "",
                    "Data directory with BSDS500 tfrecords")


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
  cfg = vgg_16_hed_config(add_v1net_early=FLAGS.add_v1net_early,
                          add_v1net=FLAGS.add_v1net,
                          cam_net=FLAGS.add_cam)
  vgg = VGG(cfg)
  predictions, endpoints = vgg.build_model(images=features["image"],
                                           cams=features["cam"],
                                           is_training=training,
                                           preprocess=FLAGS.preprocess)
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

  loss_fuse = weighted_ce_bdcn(logits=predictions,
                               labels=labels["label"],
                               gamma=FLAGS.label_gamma,
                               lbda=FLAGS.label_lbda)
  loss_side = weighted_ce_bdcn(logits=side_predictions,
                               labels=side_labels,
                               gamma=FLAGS.label_gamma,
                               lbda=FLAGS.label_lbda)
  loss = 0.5 * loss_side + 1.1 * loss_fuse + FLAGS.weight_decay * tf.reduce_mean( 
               [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                           if 'normalization' not in v.name and 'bias' not in v.name])
  
  if training:
    global_step = tf.train.get_global_step()
    steps_per_epoch = params["num_train_steps_per_epoch"]
    learning_rate = build_learning_rate(FLAGS.learning_rate,
                                        global_step,
                                        steps_per_epoch,
                                        decay_factor=0.1,
                                        decay_steps=10000)
    fast_start = min(FLAGS.learning_rate*100, 5e-4)
    fast_learning_rate = build_learning_rate(fast_start,
                                             global_step,
                                             steps_per_epoch,
                                             decay_factor=0.1,
                                             decay_steps=10000)
    slow_vars = [var for var in vgg.model_vars 
                    if "v1net" not in var.name]
    fast_vars = list(set(vgg.model_vars).difference(set(slow_vars)))
    train_op = get_optimizer(loss, learning_rate,
                             vars=slow_vars,
                             opt=FLAGS.optimizer,
                             use_tpu=FLAGS.use_tpu)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # train_op = optimizer.minimize(loss, global_step, var_list=slow_vars)

    if FLAGS.v1_timesteps:
      fast_train_op = get_optimizer(loss, 
                                    fast_learning_rate,
                                    vars=fast_vars,
                                    opt=FLAGS.optimizer,
                                    use_tpu=FLAGS.use_tpu)
      # fast_train_op = fast_optimizer.minimize(loss, global_step, var_list=fast_vars)
      train_op = tf.group([train_op, update_ops, fast_train_op])
    else:
      train_op = tf.group([train_op, update_ops])

    gs_t = tf.reshape(global_step, [1])
    lr_t = tf.reshape(learning_rate, [1])
    fast_lr_t = tf.reshape(fast_learning_rate, [1])
    loss_t = tf.reshape(loss, [1])
    loss_side_t = tf.reshape(loss_side, [1])
    loss_fuse_t = tf.reshape(loss_fuse, [1])
    imgs_t = features["image"]
    labels_t = labels["label"]
    preds_t = tf.nn.sigmoid(predictions)

    def host_call_fn(gs, lr, fast_lr, loss, loss_side, loss_fuse, 
	             imgs, lbls, preds):
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
          tf.compat.v2.summary.image('training/images',imgs,step=gs)
          tf.compat.v2.summary.image('training/predictions',1-preds,step=gs)
          return tf.summary.all_v2_summary_ops()

    host_call_args = [gs_t, lr_t, fast_lr_t, loss_t, loss_side_t, 
                      loss_fuse_t, imgs_t, labels_t, preds_t]
    host_call = (host_call_fn, host_call_args)

  if mode == tf.estimator.ModeKeys.EVAL:
    # Define evaluation metrics:
    def metric_fn(labels, logits):
      xent = tf.nn.sigmoid_cross_entropy_with_logits
      xentropy = tf.reduce_mean(xent(logits=logits,
                                  labels=labels,
                                  ))
      rmse = tf.metrics.root_mean_squared_error(labels=labels,
                                                predictions=logits)
      return {
              'xentropy': xentropy,
              'rmse': rmse,
              }
    eval_metrics = (metric_fn, [labels["label"], predictions])
    
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
                               data_dir=params["data_dir"],
                               image_size=(FLAGS.image_size,
                                           FLAGS.image_size))
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
  args = FLAGS.flag_values_dict()

  # Setting paths
  gcs_root = "gs://v1net-tpu-bucket/"
  gcs_path = "gs://v1net-tpu-bucket/%s/" % args["base_dir"]
  model_dir = os.path.join(gcs_path, "model_dir_%s" % args["experiment_name"])
  summaries_dir = os.path.join(gcs_path, "summaries")
  if not tf.gfile.Exists(model_dir):
    tf.gfile.MakeDirs(model_dir)
  if not tf.gfile.Exists(model_dir):
    tf.gfile.MakeDirs(summaries_dir)
  args["model_dir"] = model_dir
  args["summaries_dir"] = summaries_dir

  # Set training variables
  rand_seed = np.random.randint(10000)
  tf.set_random_seed(rand_seed)
  args["random_seed"] = rand_seed
  warm_start_settings = None
  args["data_dir"] = os.path.join(gcs_root, args["data_dir"])

  num_train_examples, input_fn_train = get_input_fn_train(args)
  num_val_examples, input_fn_val = get_input_fn_validation(args)

  args["num_train_examples"] = num_train_examples * args["num_epochs"]
  args["num_train_steps"] = args["num_train_examples"] // args["train_batch_size"]
  num_train_steps = args["num_train_steps"]
  num_train_steps_per_epoch = num_train_steps // args["num_epochs"]
  args["num_train_steps_per_epoch"] = num_train_steps_per_epoch

  args["num_val_examples"] = num_val_examples
  args["num_val_steps"] = args["num_val_examples"] // args["eval_batch_size"]
  num_val_steps = args["num_val_steps"]
  eval_every = int(args["evaluate_every"] * num_train_steps_per_epoch)
  
  warm_start_settings = tf.estimator.WarmStartSettings(
                                        ckpt_to_initialize_from=args['checkpoint'],
                                        vars_to_warm_start=["^(?!.*side_output|.*cam|.*v1net|.*Momentum|global_step|beta*|gamma*|.*Adam)"],
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

  # Create classifier for training
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
  try:
    current_step = tf.train.load_variable(model_dir,
                                          tf.GraphKeys.GLOBAL_STEP)
  except (TypeError, ValueError, tf.errors.NotFoundError):
    current_step = 0
  
  while current_step < num_train_steps:
    num_train_steps = min(num_train_steps, FLAGS.max_steps)
    tf.logging.info("Training for %s steps" % num_train_steps)
    if args["train_and_eval"]:
      next_checkpoint = min(num_train_steps, current_step + eval_every)
    else:
      next_checkpoint = num_train_steps
    classifier.train(
          input_fn=input_fn_train, max_steps=next_checkpoint)
    current_step = next_checkpoint
    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))
    eval_results = classifier.evaluate(input_fn=input_fn_val,
                                       steps=num_val_steps)
    tf.logging.info('Eval results at step %d: %s',
                     next_checkpoint, eval_results)

  elapsed_time = int(time.time() - start_timestamp)
  tf.logging.info("Finished training up to step %d. Elapsed seconds %d.",
                          num_train_steps, elapsed_time)

if __name__=="__main__":
  tf.logging.set_verbosity('INFO')
  app.run(main)
