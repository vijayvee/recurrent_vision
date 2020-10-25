#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-4
NUM_EPOCHS=50
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
EXPERIMENT_NAME="bsds_vgg_hed_training_ft_fin"
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
OPTIMIZER="adam"
USE_TPU=True
ADD_V1NET_EARLY=True
TPU_NAME=$1
DATA_DIR="bsds_data/HED-BSDS/tfrecords"

echo "Running on {$1}"
python bsds_main.py \
	--learning_rate=${LEARNING_RATE} \
	--weight_decay=${WEIGHT_DECAY} \
	--num_epochs=${NUM_EPOCHS} \
	--train_batch_size=${TRAIN_BATCH_SIZE} \
	--eval_batch_size=${EVAL_BATCH_SIZE} \
	--experiment_name=${EXPERIMENT_NAME} \
	--checkpoint=${CHECKPOINT} \
	--optimizer=${OPTIMIZER} \
	--use_tpu=${USE_TPU} \
	--add_v1net_early=${ADD_V1NET_EARLY} \
	--tpu_name=${TPU_NAME} \
	--data_dir=${DATA_DIR}
