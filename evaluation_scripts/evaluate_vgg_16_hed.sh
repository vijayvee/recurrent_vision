#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src:/home/vveeraba/src/recurrent_vision
cd /home/vveeraba/src/recurrent_vision;

MODEL_NAME="vgg_16_hed"
IN_DIR="/home/vveeraba/src/v1net_bsds/data/BSDS500/data/images/test"
OUT_DIR="/mnt/cube/projects/bsds500/test_predictions"
CHECKPOINT_DIR="gs://v1net-tpu-bucket/bsds_experiments/model_dir_bsds_vgg_hed_fuseconv_01_momentum_wtdecay_lrdecay_classify_post_resize"

python evaluator.py \
      --model_name=${MODEL_NAME} \
      --in_dir=${IN_DIR} \
      --out_dir=${OUT_DIR} \
      --checkpoint_dir=${CHECKPOINT_DIR}