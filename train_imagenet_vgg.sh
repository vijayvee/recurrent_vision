#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

LEARNING_RATE=1e-6
WEIGHT_DECAY=1e-4
NUM_EPOCHS=15
NUM_CORES=128
IMAGE_SIZE=224
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=256
EVALUATE_EVERY=10
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
OPTIMIZER="momentum"
MODEL_NAME="vgg_16"
EXPERIMENT_NAME="imagenet_vgg_v1net_batch_${TRAIN_BATCH_SIZE}_opt_${OPTIMIZER}_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}"
USE_TPU=True
ADD_V1NET_EARLY=True
ADD_V1NET=True
TPU_NAME=$1


echo "Running ${EXPERIMENT_NAME} on ${1}"
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
	--add_v1net=${ADD_V1NET} \
	--tpu_name=${TPU_NAME} \
	--num_cores=${NUM_CORES}
