#!/usr/bin/env bash
set -euo pipefail

datasets=("action" "commonsense" "count")
mask_methods=("Fisher" "ours") #Fisher:PIECE-F,ours:PIECE-S

base_model="/data2/TAP/model/Qwen3-VL-4B-Instruct"
save_root="/data1/TAP/model_con_mllm/1029"
epochs=5
seed=42
top_ratio=0.001

for mask_methode in ${mask_methods[@]}; do
    name_chain=""
    model_for_loop=${base_model}   # 每组mask方法都从基模型开始，不互相覆盖

    for idx in ${!datasets[@]}; do
        target=${datasets[$idx]}
        if [ -z $name_chain ]; then
            name_chain=${target}
        else
            name_chain=${name_chain}_${target}
        fi

        output_model=${save_root}/${mask_methode}_${name_chain}_epoch${epochs}_Llama3Exp_${top_ratio}
        if [ -d ${output_model}/${epochs} ]; then
            echo "训练结束，循环继续"
            model_for_loop=${output_model}/${epochs}
            continue
        fi
        if [ ! -d ${output_model} ]; then
            mkdir ${output_model}
        fi

        if [ "$mask_methode" != "Full" ]; then
            python /data2/TAP/code/PIECE_mllm_qwen/param_protection/process_mask_file.py \
                --data_path /data2/TAP/data/mlm_new/VQA-v2/train_dataset_sample_1000/${target} \
                --model_name_or_path ${model_for_loop} \
                --max_prompt_len 64 \
                --max_ans_len 32 \
                --seed ${seed} \
                --mask_method ${mask_methode} \
                --top_ratio ${top_ratio} \
                --target_name ${target}
        fi

        cp /data2/TAP/code/PIECE_mllm_qwen/scripts/train_seq_naive_back_test_new2.sh ${output_model}

        port=$(shuf -i25000-30000 -n1)

        echo "=============================="
        echo " mask_methode: ${mask_methode}"
        echo " 开始训练: ${target}"
        echo " 使用基模型: ${model_for_loop}"
        echo " 输出到: ${output_model}"
        echo "=============================="

        deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py  \
            --data_path /data2/TAP/data/mlm_new/VQA-v2/train_dataset_sample_1000/${target} \
            --model_name_or_path ${model_for_loop} \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --max_prompt_len 64 \
            --max_ans_len 32 \
            --learning_rate 1e-5 \
            --weight_decay 0. \
            --num_train_epochs ${epochs} \
            --gradient_accumulation_steps 8 \
            --lr_scheduler_type cosine \
            --num_warmup_steps 0 \
            --seed ${seed} \
            --zero_stage 2 \
            --deepspeed \
            --print_loss \
            --CL_method base \
            --enable_tensorboard \
            --tensorboard_path ${output_model}/log/ \
            --offload \
            --mask_method ${mask_methode} \
            --top_ratio ${top_ratio} \
            --target_name ${target} \
            --output_dir ${output_model} \
            --test_file_dir /data2/TAP/data/TRACE-Benchmark/LLM-CL-Benchmark_1000 \
            | tee ${output_model}/train.log

        model_for_loop=${output_model}/${epochs}
    done
done
