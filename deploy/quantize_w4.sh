#!/usr/bin/bash

MODEL_PATH=$1
lmdeploy lite auto_awq \
  --model  ${MODEL_PATH} \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_output

lmdeploy convert ${MODEL_PATH} ./quant_output \
    --model-format awq \
    --group-size 128