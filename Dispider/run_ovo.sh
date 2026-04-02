#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval_ovo_dispider.py \
    --model-path /你的/Dispider模型路径 \
    --data-path /你的/ovo.json路径 \
    --video-root /你的/Ego4D根目录 \
    --output ./ovo_results.json