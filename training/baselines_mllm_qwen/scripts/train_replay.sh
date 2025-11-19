#!/bin/bash
for target in "replay" "replay_online"; do
    output_dir=/data2/TAP/model_con/baseline/${target}
    # 需要修改到自己的输入目录
    if [ ! -d ${output_dir} ];then  
        mkdir ${output_dir}
    fi
    learning_rate=1e-5
    if [ "${target}" = "O-LoRA" ]; then
        learning_rate=1e-4
    fi
    port=$(shuf -i25000-30000 -n1)
    deepspeed --include=localhost:0,1,2,3 --master_port $port training/${target}.py  \
        --data_path /data2/TAP/data/TRACE-Benchmark/LLM-CL-Benchmark_1000 \
        --dataset_name C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten,Py150 \
        --model_name_or_path /data2/TAP/model/Meta-Llama-3-8B \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --max_prompt_len 1024 \
        --max_ans_len 512 \
        --learning_rate 1e-5 \
        --weight_decay 0. \
        --num_train_epochs 5,5,5,5,5,5,5,5 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed 42 \
        --zero_stage 2 \
        --offload \
        --deepspeed \
        --print_loss \
        --past_task_ratio 0.1 \
        --ood_eval_dir /data2/TAP/data/continue_eval_loss_data_small \
        --output_dir ${output_dir} \
        | tee ${output_dir}/train.log
done


# for slurm running
# srun --partition=xai --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51459 training/replay.py  \
#     --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
#     --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --replay_dataset_name Lima \
#     --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/llama2HF/7B-Chat \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 16 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --num_train_epochs 5,3,7,5,3,5,5,7 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --num_warmup_steps 0 \
#     --seed 1234 \
#     --zero_stage 3 \
#     --deepspeed \
#     --print_loss \
#     --past_task_ratio 0.1 \
#     --output_dir /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/replay/llama7b-chat > llama2-7b-replay.log 2>&1 &



# srun --partition=xai --mpi=pmi2 --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=asb-llama --kill-on-bad-exit=1 /mnt/petrelfs/wangxiao/miniconda3/envs/cl/bin/deepspeed --master_port 51459 training/replay.py  \
#     --data_path /mnt/petrelfs/wangxiao/DATA/LLM-CL-Benchmark/LLM-CL-Benchmark_5000 \
#     --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
#     --replay_dataset_name Lima \
#     --model_name_or_path /mnt/petrelfs/wangxiao/MODELS/vicuna-7b-v1.5 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 16 \
#     --max_prompt_len 1024 \
#     --max_ans_len 512 \
#     --learning_rate 1e-5 \
#     --weight_decay 0. \
#     --num_train_epochs 5,3,7,5,3,5,5,7 \
#     --gradient_accumulation_steps 16 \
#     --lr_scheduler_type cosine \
#     --num_warmup_steps 0 \
#     --seed 1234 \
#     --zero_stage 3 \
#     --deepspeed \
#     --print_loss \
#     --past_task_ratio 0.1 \
#     --output_dir /mnt/petrelfs/wangxiao/LLM-Continual-Learning/CKPT/replay/vicuna-7b > vicuna-7b-replay.log 2>&1 &
