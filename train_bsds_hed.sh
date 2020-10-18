#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
NUM_EPOCHS=100
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
EVALUATE_EVERY=2
EXPERIMENT_NAME="bsds_vgg_hed_test"
OPTIMIZER="adam"
USE_TPU=True
TPU_NAME="test-tpu"
DATA_DIR="bsds_data/bsds500"

python bsds_main.py \
	--learning_rate=${LEARNING_RATE} \
	--weight_decay=${WEIGHT_DECAY} \
	--num_epochs=${NUM_EPOCHS} \
	--train_batch_size=${TRAIN_BATCH_SIZE} \
	--eval_batch_size=${EVAL_BATCH_SIZE} \
	--evaluate_every=${EVALUATE_EVERY} \
	--experiment_name=${EXPERIMENT_NAME} \
	--optimizer=${OPTIMIZER} \
	--use_tpu=${USE_TPU} \
	--tpu_name=${TPU_NAME} \
	--data_dir=${DATA_DIR}
