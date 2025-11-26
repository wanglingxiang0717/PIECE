#!/usr/bin/env bash
output_model=/data2/TAP/model_exp/1125_test_deepspeed_4_all_new
port=$(shuf -i25000-30000 -n1)
if [ ! -d ${output_model} ]; then
    mkdir ${output_model}
fi
deepspeed --include=localhost:0,1,2,3 --master_port $port training/torch_train_main_deepspeed.py \
    --data_path example_data/LLM-CL-Benchmark_500/C-STANCE \
    --model_name_or_path /data2/TAP/model/Meta-Llama-3-8B \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 42 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --enable_tensorboard \
    --tensorboard_path ${output_model}/log/ \
    --offload \
    --top_ratio 0.001 \
    --output_dir ${output_model} \
    --param_import_savepath ${output_model} \
    | tee ${output_model}/train.log