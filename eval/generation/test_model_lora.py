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


def main():
    datasets = ["C-STANCE", 
                "FOMC", 
                "MeetingBank", 
                "ScienceQA", 
                "NumGLUE-cm", 
                "NumGLUE-ds", 
                "20Minuten", 
                "Py150"]

    # for 
    model_path = "/data/TAP/wanglingxiang/wanglingxiang/model/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    
    for dataset in datasets[-1:]:
        # stop = input(dataset)
        # continue
        for i in range(5):
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
            lora_model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/lora_1024/{dataset}/{i}"
            model = PeftModel.from_pretrained(base_model, lora_model_path, device_map={"": 0})
    # tokenizer.truncation_side = "left"
            batch_size = 1
            index = datasets.index(dataset)
            
            # batch_size = 2
            for test_data in datasets[:index+1]:
                if test_data in ["C-STANCE", 
                    "FOMC", 
                    "NumGLUE-cm", 
                    "NumGLUE-ds"]:
                    batch_size = 8
                    max_len = 16
                else:
                    batch_size = 4
                    max_len = 256
                file_path = f"/data/TAP/wanglingxiang/wanglingxiang/LLM-CL-Benchmark_1000/{test_data}/test.json"
                file_save_path = f"{lora_model_path}/{test_data}_multibatch_result.json"
                if not os.path.exists(file_save_path): 
                    data = read_json_or_jsonl(file_path)
                    completion_dict = []

                    for i in tqdm(range(0, len(data), batch_size)):
                        batch_items = data[i : i + batch_size]
                        batch_prompts = [item["prompt"] for item in batch_items]

                        generations = generate_batch(batch_prompts, model, tokenizer, max_len=max_len)

                        for item, gen in zip(batch_items, generations):
                            result_json = {
                                "input": str(item["prompt"]).strip(),
                                "label": str(item["answer"]).strip(),
                                "model_output": gen,
                                "filtered_output": gen,
                            }
                            completion_dict.append(result_json)

                    write_json_or_jsonl(file_save_path, completion_dict)
                # print(f"{target} done, results saved to {file_save_path}")


if __name__ == "__main__":
    main()
