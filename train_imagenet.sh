#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

LEARNING_RATE=1e-6
WEIGHT_DECAY=1e-4
NUM_EPOCHS=15
NUM_CORES=128
IMAGE_SIZE=299
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=256
EVALUATE_EVERY=10
EXPERIMENT_NAME="imagenet_resnet_batch256_img299_v1net64_complexcell"
MODEL_NAME="resnet_v2_50"
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/resnet_v2_50/resnet_v2_50.ckpt"
OPTIMIZER="momentum"
USE_TPU=True
ADD_V1NET_EARLY=True
COMPACT=False
TPU_NAME=$1

echo "Running on {$1}"
python main_imagenet.py \
	--learning_rate=${LEARNING_RATE} \
	--weight_decay=${WEIGHT_DECAY} \
	--num_epochs=${NUM_EPOCHS} \
	--image_size=${IMAGE_SIZE} \
	--train_batch_size=${TRAIN_BATCH_SIZE} \
	--eval_batch_size=${EVAL_BATCH_SIZE} \
	--evaluate_every=${EVALUATE_EVERY} \
	--experiment_name=${EXPERIMENT_NAME} \
	--model_name=${MODEL_NAME} \
	--checkpoint=${CHECKPOINT} \
	--optimizer=${OPTIMIZER} \
	--use_tpu=${USE_TPU} \
	--add_v1net_early=${ADD_V1NET_EARLY} \
	--compact=${COMPACT} \
	--tpu_name=${TPU_NAME} \
	--num_cores=${NUM_CORES}
