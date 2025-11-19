from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
from collections import defaultdict
import transformers
import torch
import csv
import os
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm
import jsonlines
import json
import re
# from find_simple_tool import get_filter
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
    is_torch_tpu_available,
    set_seed,
    DataCollatorForSeq2Seq
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

from training.train_base_model import CL_Base_Model

# from params import Method2Class, AllDatasetName
replace_llama_attn_with_flash_attn()
replace_bloom_attn_with_flash_attn()

cuda_device_index = 0
cuda_device_index_deepspeed = f'localhost:{cuda_device_index}'

# csv_file = '/home/TAP/wanglingxiang/Data_Organization/data/train_data/split_all.csv' 
file_A = '/data/TAP/wanglingxiang/wanglingxiang/data_backup/Data_Organization/data' 

output_model = '/data/TAP/wanglingxiang/wanglingxiang/Data_Organization-workspace/model/Data_annotation_test_1'
# output_dir = '/data/TAP/wanglingxiang/wanglingxiang/Data_Organization-workspace/cache'

def data_loader(tokenizer, dataset_path=None, train_dataset_save=None):
    # dataset_path = data_path
    # Prepare the data
    if train_dataset_save is None:
        train_dataset, _, _ = create_prompt_dataset(
            -1,
            dataset_path,
            "/tmp/data_files/",
            42,
            distributed=False
        )

    else:
        train_dataset = train_dataset_save
    # DataLoaders creation:
    train_sampler = SequentialSampler(train_dataset)

    data_collator = DataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=2048,
        max_ans_len=512,
        pad_to_multiple_of=8,
        inference=False
    )       

    train_dataloader = DataLoader(train_dataset,
                                collate_fn=data_collator,
                                sampler=train_sampler,
                                batch_size=1)
    if train_dataset_save is None:
        return train_dataloader, train_dataset
    else:
        return train_dataloader
    
def create_output_directory():
    if not os.path.exists(output_model):
        os.makedirs(output_model)
    # shutil.copy('finetune.sh', output_model)

def clean_up():
    if os.path.exists(output_model):
        shutil.rmtree(output_model)
    if os.path.exists(file_A):
        # shutil.rmtree(file_A)
        os.remove(os.path.join(file_A, 'train.json'))
        os.remove(os.path.join(file_A, 'eval.json'))
        os.remove(os.path.join(file_A, 'test.json'))
    torch.cuda.empty_cache()

#标注任务主体
def test_model(index, train_dataset, loss_list, tokenizer, loss_sum):
    
    assert train_dataset[index] == loss_list[index]['row_data']
    
    model = create_hf_model(AutoModelForCausalLM,
                        os.path.join(output_model, "1"),
                        tokenizer,
                        cuda_device_index=cuda_device_index
                        )
    model.eval()

    loss_list_index = []
    train_dataloader = data_loader(tokenizer, train_dataset_save=train_dataset)
    for en, batch in enumerate(tqdm(train_dataloader)):
    # print(batch['sources'])
    # stop = input()
        loss_dict_index = {}
        loss_dict_index['row_data'] = train_dataset[en]
        del batch['sources']
        with torch.no_grad():
            batch = to_device(batch, model.device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss_dict_index['loss'] = loss.item()
            loss_list_index.append(loss_dict_index)
            del loss, outputs, batch
            torch.cuda.empty_cache()

    assert len(loss_list_index) == len(loss_list)
    loss_sum_index = sum(float(loss_list_index[i]['loss']) for i in range(len(loss_list_index)))
    loss_change_all = loss_sum - loss_sum_index
    
    # for i in range(len(loss_list)):
    #     if i == index:
    #         continue
    #     if float(loss_list[i]['loss']) >= float(loss_list_index[i]['loss']):
    #         save_data.append(loss_list_index[i]['row_data'])
    with open(f'/data/TAP/wanglingxiang/wanglingxiang/data_sys/TRACE-Benchmark/LLM-CL-Benchmark_500/MeetingBank/loss_change/simple-{index}.csv', 'w') as file_w:
        writer = csv.writer(file_w)
        writer.writerow(['input', 'target', 'loss_change'])
        
        inputs = train_dataset[index]['prompt']
        target = train_dataset[index]['answer']
        
        simple = [inputs, target]
        save_index = simple + [loss_change_all]
        writer.writerow(save_index)
        # writer.writerow(row)
        for i in range(len(loss_list)):
            if i == index:
                continue
            loss_change = float(loss_list[i]['loss']) - float(loss_list_index[i]['loss'])
            if loss_change >= 0:
                inputs = loss_list_index[i]['row_data']['prompt']
                target = loss_list_index[i]['row_data']['answer']
                save_index = [inputs, target, loss_change]
                writer.writerow(save_index)
    
    # return hidden_embedding
    
#hidden_state以及后续的embedding放在后面进行计算

# def count_lines_in_csv(csv_file):
#     df = pd.read_csv(csv_file)
#     return len(df)

# total_lines = count_lines_in_csv(csv_file) 

def filter_original(row_model, json_file):
    tokenizer = load_hf_tokenizer(row_model, fast_tokenizer=True)
    tokenizer_dec = AutoTokenizer.from_pretrained(row_model)
    train_dataloader, train_dataset = data_loader(tokenizer, json_file)
    model = create_hf_model(AutoModelForCausalLM,
                        row_model,
                        tokenizer,
                        cuda_device_index=cuda_device_index
                        )
    model.eval()
    loss_list = []
    for en, batch in enumerate(tqdm(train_dataloader)):
        loss_dict = {}
        loss_dict['row_data'] = train_dataset[en]
        del batch['sources']
        with torch.no_grad():
            batch = to_device(batch, model.device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss_dict['loss'] = loss.item()
            loss_list.append(loss_dict)
            del loss, outputs, batch
            torch.cuda.empty_cache()
    loss_sum = sum(float(loss_list[i]['loss']) for i in range(len(loss_list)))
    
    del model 
    torch.cuda.empty_cache()

    for index in tqdm(range(len(train_dataset))):
        if index <= 1:
            continue  
        # stop = input(index)   
        data = [train_dataset[index]]  
        with open(os.path.join(file_A, 'train.json'), mode='w', newline='') as file_a:
            json.dump(data, file_a)
        with open(os.path.join(file_A, 'eval.json'), mode='w', newline='') as file_a:
            json.dump(data, file_a)
        with open(os.path.join(file_A, 'test.json'), mode='w', newline='') as file_a:
            json.dump(data, file_a)
            
        create_output_directory()
        
        command = [
            'deepspeed', '--include', 'localhost:2', '--master_port', '29506', 'training/main.py',
            '--data_path', file_A,
            '--model_name_or_path', row_model,
            '--per_device_train_batch_size', '1',
            '--per_device_eval_batch_size', '1',
            '--max_prompt_len', '2048',
            '--max_ans_len', '512',
            '--learning_rate', '1e-5', 
            '--weight_decay', '0.',
            '--num_train_epochs', '1',
            '--gradient_accumulation_steps', '1',
            '--lr_scheduler_type', 'cosine',
            '--num_warmup_steps', '0',
            '--seed', '42',
            '--zero_stage', '2',
            '--deepspeed',
            '--print_loss',
            '--CL_method', 'base', 
            '--offload',
            '--output_dir', output_model,
        ]

        with open(f'{output_model}/train.log', 'a') as log_file:
            subprocess.run(command, stdout=log_file)
        
        test_model(index, train_dataset, loss_list, tokenizer, loss_sum)
        
        clean_up()

    # print(my_dict)
    # print(get_filter(my_dict))
    # return get_filter(my_dict)

if __name__ == "__main__":
    json_file = '/data/TAP/wanglingxiang/wanglingxiang/data_sys/TRACE-Benchmark/LLM-CL-Benchmark_500/MeetingBank/' 
    row_model = '/data/TAP/wanglingxiang/wanglingxiang/model/Llama-2-7b-chat-hf'
    filter_original(row_model, json_file)

    