#!/usr/bin/bash

MODEL_PATH=$1

# lmdeploy chat turbomind ${MODEL_PATH} --model-name internlm-chat-20b
lmdeploy convert internlm-chat-7b $1
lmdeploy chat turbomind ./workspace