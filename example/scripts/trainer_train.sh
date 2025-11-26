#!/bin/bash
OUTPUT_DIR="/data2/TAP/model_exp/1125_deepspeed_zero2_4gpu_full"

deepspeed --num_gpus=4 training/trainer_train_main_deepspeed.py \
    --model_name_or_path /data2/TAP/model/Meta-Llama-3-8B \
    --param_import_savepath ${OUTPUT_DIR} \
    --top_ratio 0.001 \
    --data_dir example_data/LLM-CL-Benchmark_500/C-STANCE \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --lr 1e-5 \
    --num_epochs 1 \
    --gradient_accumulation_steps 1
