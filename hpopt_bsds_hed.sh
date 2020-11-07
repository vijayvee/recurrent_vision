#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

NUM_EPOCHS=10
LEARNING_RATE=$2
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
OPTIMIZER="adam"
EVALUATE_EVERY=1
BASE_DIR="bsds_hpopt_multiv1net"
USE_TPU=True
ADD_V1NET_EARLY=True
ADD_V1NET=True
TPU_NAME=$1
PREPROCESS=True
NUM_CORES=8
DATA_DIR="bsds_data/HED-BSDS/tfrecords"

declare -a wd=(5e-4 2e-4 1e-4)
for WEIGHT_DECAY in "${wd[@]}"
do
    EXPERIMENT_NAME="hed_multiv1net_hpopt_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}_opt_${OPTIMIZER}_preprocess_${PREPROCESS}_traineval_mse"
    echo "Running ${EXPERIMENT_NAME} on ${1}"
    python main_bsds_traineval.py \
       --learning_rate=${LEARNING_RATE} \
       --weight_decay=${WEIGHT_DECAY} \
       --num_epochs=${NUM_EPOCHS} \
       --train_batch_size=${TRAIN_BATCH_SIZE} \
       --eval_batch_size=${EVAL_BATCH_SIZE} \
       --experiment_name=${EXPERIMENT_NAME} \
       --checkpoint=${CHECKPOINT} \
       --optimizer=${OPTIMIZER} \
       --evaluate_every=${EVALUATE_EVERY} \
       --base_dir=${BASE_DIR} \
       --use_tpu=${USE_TPU} \
       --add_v1net_early=${ADD_V1NET_EARLY} \
       --add_v1net=${ADD_V1NET} \
       --tpu_name=${TPU_NAME} \
       --preprocess=${PREPROCESS} \
       --num_cores=${NUM_CORES} \
       --data_dir=${DATA_DIR}  
done
