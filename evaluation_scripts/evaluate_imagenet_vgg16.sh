#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/vveeraba/src:/home/vveeraba/src/recurrent_vision
cd /home/vveeraba/src/recurrent_vision;

MODEL_NAME="vgg_16"
IN_DIR="gs://v1net-tpu-bucket/imagenet_data/"
OUT_DIR="gs://v1net-tpu-bucket/checkpoints/vgg_16"
CHECKPOINT_DIR="models/pretrained_nets/checkpoints/vgg_16/vgg_16.ckpt"
PREPROCESS=True

python evaluator.py \
      --model_name=${MODEL_NAME} \
      --in_dir=${IN_DIR} \
      --out_dir=${OUT_DIR} \
      --checkpoint_dir=${CHECKPOINT_DIR} \
      --preprocess=${PREPROCESS}
