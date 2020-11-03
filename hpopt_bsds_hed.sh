#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

NUM_EPOCHS=10
TRAIN_BATCH_SIZE=128
EVAL_BATCH_SIZE=128
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
OPTIMIZER="adam"
BASE_DIR="bsds_hpopt"
USE_TPU=True
ADD_V1NET_EARLY=True
TPU_NAME=$1
NUM_CORES=128
DATA_DIR="bsds_data/HED-BSDS/tfrecords"

declare -a lr=(5e-4 1e-4 5e-5 1e-5)
declare -a wd=(5e-4 1e-4 5e-5 1e-5)
for LEARNING_RATE in "${lr[@]}"
  do
  for WEIGHT_DECAY in "${wd[@]}"
    do
    EXPERIMENT_NAME="hed_v1net_hpopt_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}"
    echo "Running ${EXPERIMENT_NAME} on ${1}"
    # python main_bsds.py \
    #   --learning_rate=${LEARNING_RATE} \
    #   --weight_decay=${WEIGHT_DECAY} \
    #   --num_epochs=${NUM_EPOCHS} \
    #   --train_batch_size=${TRAIN_BATCH_SIZE} \
    #   --eval_batch_size=${EVAL_BATCH_SIZE} \
    #   --experiment_name=${EXPERIMENT_NAME} \
    #   --checkpoint=${CHECKPOINT} \
    #   --optimizer=${OPTIMIZER} \
    #    --base_dir=${BASE_DIR} \
    #   --use_tpu=${USE_TPU} \
    #   --add_v1net_early=${ADD_V1NET_EARLY} \
    #   --tpu_name=${TPU_NAME} \
    #   --num_cores=${NUM_CORES} \
    #   --data_dir=${DATA_DIR}  