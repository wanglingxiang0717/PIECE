#!/bin/bash
for target in "base" "EWC" "GEM" "lora" "LwF" "MIGU" "Norm" "O-LoRA"; do
    cl_method=${target}
    port=$(shuf -i25000-30000 -n1)
    output_dir=/data1/TAP/model_mlm_save_new_Qwen3/${cl_method}
    # 需要修改到自己的输入目录
    if [ ! -d ${output_dir} ];then  
        mkdir ${output_dir}
    fi
    learning_rate=1e-5
    if [ "${target}" = "O-LoRA" ] || [ "${target}" = "lora" ]; then
        learning_rate=1e-4
    fi
    deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py  \
        --data_path /data2/TAP/data/mlm_new/VQA-v2/train_dataset_sample_1000 \
        --dataset_name action,commonsense,count \
        --model_name_or_path /data2/TAP/model/Qwen3-VL-4B-Instruct \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --max_prompt_len 64 \
        --max_ans_len 32 \
        --learning_rate ${learning_rate} \
        --weight_decay 0. \
        --num_train_epochs 5,5,5 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 42 \
        --zero_stage 2 \
        --deepspeed \
        --print_loss \
        --CL_method ${cl_method} \
        --offload \
        --test_file_dir /data2/TAP/data/mlm_new/VQA-v2/Partition_Q \
        --mask_save_file_dir ${output_dir} \
        --mask_top_ratio 0.001 \
        --output_dir ${output_dir} \
        | tee ${output_dir}/train.log
done

for target in "replay_online"; do
    cl_method=${target}
    port=$(shuf -i25000-30000 -n1)
    output_dir=/data1/TAP/model_mlm_save_new_Qwen3/${cl_method}
    # 需要修改到自己的输入目录
    if [ ! -d ${output_dir} ];then  
        mkdir ${output_dir}
    fi
    learning_rate=1e-5
    if [ "${target}" = "O-LoRA" ]; then
        learning_rate=1e-4
    fi
    deepspeed --include=localhost:0,1,2,3 --master_port $port training/${target}.py  \
        --data_path /data2/TAP/data/mlm_new/VQA-v2/train_dataset_sample_200 \
        --dataset_name action,commonsense \
        --model_name_or_path /data2/TAP/model/Qwen3-VL-4B-Instruct \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --max_prompt_len 64 \
        --max_ans_len 32 \
        --learning_rate ${learning_rate} \
        --weight_decay 0. \
        --num_train_epochs 5,5,5 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 42 \
        --zero_stage 2 \
        --offload \
        --deepspeed \
        --print_loss \
        --past_task_ratio 0.1 \
        --output_dir ${output_dir} \
        | tee ${output_dir}/train.log
done

# for target in "SPTCF"; do
#     cl_method=${target}
#     port=$(shuf -i25000-30000 -n1)
#     output_dir=/data1/TAP/model_con/baseline_14B/${cl_method}
#     # 需要修改到自己的输入目录
#     if [ ! -d ${output_dir} ];then  
#         mkdir ${output_dir}
#     fi
#     learning_rate=1e-5
#     if [ "${target}" = "O-LoRA" ]; then
#         learning_rate=5e-4
#     fi
#     deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py  \
#         --data_path /data2/TAP/data/TRACE-Benchmark/LLM-CL-Benchmark_50 \
#         --dataset_name C-STANCE,FOMC \
#         --data_output_path ${output_dir} \
#         --model_name_or_path /data2/TAP/model/Qwen/Qwen3-14B \
#         --per_device_train_batch_size 1 \
#         --per_device_eval_batch_size 1 \
#         --max_prompt_len 256 \
#         --max_ans_len 128 \
#         --learning_rate ${learning_rate} \
#         --weight_decay 0. \
#         --num_train_epochs 1,1,1,1,1,1,1,1 \
#         --gradient_accumulation_steps 16 \
#         --lr_scheduler_type cosine \
#         --num_warmup_steps 0 \
#         --seed 42 \
#         --zero_stage 2 \
#         --deepspeed \
#         --print_loss \
#         --CL_method ${cl_method} \
#         --offload \
#         --test_file_dir /data2/TAP/data/TRACE-Benchmark/LLM-CL-Benchmark_50 \
#         --mask_save_file_dir ${output_dir} \
#         --mask_top_ratio 0.01 \
#         --output_dir ${output_dir} \
#         | tee ${output_dir}/train.log
# done

