#!bin/bash
cl_method="O-LoRA"
port=$(shuf -i25000-30000 -n1)
output_dir=/data2/TAP/model_con/${cl_method}
# 需要修改到自己的输入目录
if [ ! -d ${output_dir} ];then  
    mkdir ${output_dir}
fi
deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py  \
    --data_path /data2/TAP/data/TRACE-Benchmark/LLM-CL-Benchmark_1000 \
    --dataset_name C-STANCE,FOMC,MeetingBank,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten,Py150 \
    --model_name_or_path /data2/TAP/model/Meta-Llama-3-8B \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --num_train_epochs 5,5,5,5,5,5,5,5 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 42 \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method ${cl_method} \
    --offload \
    --ood_eval_dir /data2/TAP/data/continue_eval_loss_data_small \
    --output_dir ${output_dir} \
    | tee ${output_dir}/train.log

