#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

NUM_EPOCHS=50
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
V1_TIMESTEPS=$2
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
OPTIMIZER="adam"
EVALUATE_EVERY=1
BASE_DIR="bsds_hpopt_multiv1net/hpopt3-training-timesteps-60k"
USE_TPU=True
ADD_V1NET_EARLY=True
ADD_V1NET=True
TPU_NAME=$1
PREPROCESS=True
TRAIN_AND_EVAL=False
NUM_CORES=8
DATA_DIR="bsds_data/HED-BSDS/tfrecords"

for RUN in 1 2 3
do
EXPERIMENT_NAME="hed_multiv1net_hpopt_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}_opt_${OPTIMIZER}_preprocess_${PREPROCESS}_timesteps_${V1_TIMESTEPS}_posweight_1_1_train_run_${RUN}"
echo "Running ${EXPERIMENT_NAME} on ${1}"
python main_bsds_traineval.py \
       --learning_rate=${LEARNING_RATE} \
       --weight_decay=${WEIGHT_DECAY} \
       --num_epochs=${NUM_EPOCHS} \
       --train_batch_size=${TRAIN_BATCH_SIZE} \
       --eval_batch_size=${EVAL_BATCH_SIZE} \
       --v1_timesteps=${V1_TIMESTEPS} \
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
       --train_and_eval=${TRAIN_AND_EVAL} \
       --num_cores=${NUM_CORES} \
       --data_dir=${DATA_DIR}  
done
