#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-4
NUM_EPOCHS=10
TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=64
EVALUATE_EVERY=10
EXPERIMENT_NAME="imagenet_vgg_ft_v1net_early"
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
OPTIMIZER="momentum"
USE_TPU=True
ADD_V1NET_EARLY=True
TPU_NAME=$1

echo "Running on {$1}"
python imagenet_train.py \
	--learning_rate=${LEARNING_RATE} \
	--weight_decay=${WEIGHT_DECAY} \
	--num_epochs=${NUM_EPOCHS} \
	--train_batch_size=${TRAIN_BATCH_SIZE} \
	--eval_batch_size=${EVAL_BATCH_SIZE} \
	--evaluate_every=${EVALUATE_EVERY} \
	--experiment_name=${EXPERIMENT_NAME} \
	--checkpoint=${CHECKPOINT} \
	--optimizer=${OPTIMIZER} \
	--use_tpu=${USE_TPU} \
	--add_v1net_early=${ADD_V1NET_EARLY} \
	--tpu_name=${TPU_NAME}
