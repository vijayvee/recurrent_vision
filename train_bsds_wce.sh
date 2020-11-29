#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src/recurrent_vision:/home/vveeraba/src

NUM_EPOCHS=15
LEARNING_RATE=1e-6
WEIGHT_DECAY=2e-4
LABEL_GAMMA=0.3
LABEL_LBDA=1.1
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
IMAGE_SIZE=512
V1_TIMESTEPS=$2
CHECKPOINT="gs://v1net-tpu-bucket/checkpoints/vgg_16/vgg_16.ckpt"
OPTIMIZER="bsds_momentum"
EVALUATE_EVERY=1
BASE_DIR="bsds_hpopt_multiv1net/hpopt3-training-timesteps-20k"
USE_TPU=True
ADD_V1NET_EARLY=False
ADD_V1NET=False
TPU_NAME=$1
PREPROCESS=True
TRAIN_AND_EVAL=False
NUM_CORES=8
DATA_DIR="bsds_data/HED-BSDS/cam_tfrecords_float_gt"

# wce_custom runs use the weighted_ce loss in losses.py
EXPERIMENT_NAME="randaug_modulat_hed_wce_side_0_5_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}_gamma_${LABEL_GAMMA}_lbda_${LABEL_LBDA}_opt_${OPTIMIZER}_preprocess_${PREPROCESS}_timesteps_${V1_TIMESTEPS}_tpu_${TPU_NAME}"
echo "Running ${EXPERIMENT_NAME} on ${1}"
python main_bsds_wce.py \
       --learning_rate=${LEARNING_RATE} \
       --weight_decay=${WEIGHT_DECAY} \
       --label_gamma=${LABEL_GAMMA} \
       --num_epochs=${NUM_EPOCHS} \
       --train_batch_size=${TRAIN_BATCH_SIZE} \
       --eval_batch_size=${EVAL_BATCH_SIZE} \
       --image_size=${IMAGE_SIZE} \
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
