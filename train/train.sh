#!/bin/bash

python -m accelerate.launch \
    --config_file config.yaml \
    train.py \
    --model_name "../../models/Qwen3-4B" \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --max_length 2048 \
    --num_workers 4 \
    --train_data_path "../data/train.json" \
    --eval_data_path "../data/val.json" \
    --output_dir "../output" \
    --save_steps 100 \
    --eval_steps 100