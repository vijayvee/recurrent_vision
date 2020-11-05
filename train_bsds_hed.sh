#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
NUM_EPOCHS=50
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
OPTIMIZER="adam"
PREPROCESS=True
EXPERIMENT_NAME="hed_bs8_multiv1net_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}_opt_${OPTIMIZER}_5v1nets_preprocess_posweight_1_1"
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
USE_TPU=True
ADD_V1NET_EARLY=True
ADD_V1NET=True
TPU_NAME=$1
NUM_CORES=8
DATA_DIR="bsds_data/HED-BSDS/tfrecords"

echo "Running ${EXPERIMENT_NAME} on ${1}"
python main_bsds.py \
	--learning_rate=${LEARNING_RATE} \
	--weight_decay=${WEIGHT_DECAY} \
	--num_epochs=${NUM_EPOCHS} \
	--train_batch_size=${TRAIN_BATCH_SIZE} \
	--eval_batch_size=${EVAL_BATCH_SIZE} \
	--experiment_name=${EXPERIMENT_NAME} \
	--checkpoint=${CHECKPOINT} \
	--optimizer=${OPTIMIZER} \
	--preprocess=${PREPROCESS} \
	--use_tpu=${USE_TPU} \
	--add_v1net_early=${ADD_V1NET_EARLY} \
	--add_v1net=${ADD_V1NET} \
	--tpu_name=${TPU_NAME} \
	--num_cores=${NUM_CORES} \
	--data_dir=${DATA_DIR}
