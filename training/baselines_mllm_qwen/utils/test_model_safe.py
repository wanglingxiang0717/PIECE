# from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
import transformers
import torch
import csv
import os
import subprocess
import shutil
import pandas as pd
import jsonlines
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel,PeftConfig
import torch.distributed as dist

def read_json_or_jsonl(file_path):
    data = []
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'r') as f_r:
            data = json.load(f_r)
    elif txt == 'jsonl':
        with open(file_path, 'r') as f_r:
            for line in f_r:
                data.append(json.loads(line))
    else:
        data = None
    return data

def write_json_or_jsonl(file_path, data, encoding='utf-8'):
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'w', encoding=encoding) as f_w:
            json.dump(data, f_w, ensure_ascii=False, indent=2)
    elif txt == 'jsonl':
        with open(file_path, 'w', encoding=encoding) as f_w:
            for line in data:
                json_line = json.dumps(line)
                f_w.write(json_line + '\n')
    else:
        print("file_path not exisits")

def generate_batch(prompts, model, tokenizer, max_len=256):
    # tokenizer.padding_side == 'left'
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024 + max_len
        ).to(model.device)
        
        sequences = model.generate(
            **inputs,
            do_sample=True,
            top_k=10,
            temperature=0.3,
            top_p=0.95,
            max_new_tokens=max_len,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            # use_cache=False,
        )

    generated_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    # 去掉 prompt 前缀，保证只保留模型生成内容
    cleaned_texts = []
    for gen, prompt in zip(generated_texts, prompts):
        if gen.startswith(prompt):
            cleaned_texts.append(gen[len(prompt):].strip())
        else:
            cleaned_texts.append(gen.strip())
    del inputs, sequences, generated_texts
    torch.cuda.empty_cache()
    return cleaned_texts

# def generation_safe_file(model, tokenizer, file_dir, file_save_dir):
#     print("*********  generation safe *************")
#     batch_size = 8
#     max_len = 256
#     file_path = f"{file_dir}/MaliciousInstruct/test.json"
#     file_save_path = f"{file_save_dir}/MaliciousInstruct_multibatch_result.json"
#     # if not os.path.exists(file_save_path): 
#     data = read_json_or_jsonl(file_path)
#     completion_dict = []

#     for i in tqdm(range(0, len(data), batch_size)):
#         batch_items = data[i : i + batch_size]
#         batch_prompts = [item["prompt"] for item in batch_items]

#         generations = generate_batch(batch_prompts, model, tokenizer, max_len=max_len)

#         for item, gen in zip(batch_items, generations):
#             result_json = {
#                 "input": str(item["prompt"]).strip(),
#                 "model_output": gen,
#                 "filtered_output": gen,
#             }
#             completion_dict.append(result_json)

#     write_json_or_jsonl(file_save_path, completion_dict) 

def generation_safe_file(model, tokenizer, file_dir):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    batch_size = 8
    max_len = 256
    file_path = f"{file_dir}/MaliciousInstruct/test.json"
    # file_save_path = f"{file_save_dir}/MaliciousInstruct_multibatch_result.json"
    # if not os.path.exists(file_save_path): 
    data = read_json_or_jsonl(file_path)
    num_samples = len(data)
        
    samples_per_rank = num_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    if rank == world_size - 1:
        end_idx = num_samples

    data_rank = data[start_idx:end_idx]
    completion_dict = []

    for i in tqdm(range(0, len(data_rank), batch_size), disable=(rank != 0)):
        batch_items = data_rank[i : i + batch_size]
        batch_prompts = [item["prompt"] for item in batch_items]
        generations = generate_batch(batch_prompts, model, tokenizer, max_len=max_len)
        for item, gen in zip(batch_items, generations):
            completion_dict.append({
                "input": str(item["prompt"]).strip(),
                "model_output": gen,
                "filtered_output": gen,
            })
    return completion_dict
