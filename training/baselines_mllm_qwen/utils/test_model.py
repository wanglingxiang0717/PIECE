# from human_eval.data import write_jsonl, read_problems
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
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, LlavaForConditionalGeneration

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

def get_conversation(question, processor):
    if question:
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}"},
                {"type": "image"},
                ],
            },
        ]
    else: 
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": f"Give a short description of the image."},
                {"type": "image"},
                ],
            },
        ]       
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt


def generation_file(dataset, datasets, model, processor, file_dir, file_save_dir):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    index = datasets.index(dataset)
    task_result_dict = {}

    for test_data in datasets[:index+1]:
        batch_size = 32

        file_path = f"{file_dir}/karpathy_test_q_{test_data}.json"
        data = read_json_or_jsonl(file_path)
        num_samples = len(data)
        
        samples_per_rank = num_samples // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank
        if rank == world_size - 1:
            end_idx = num_samples

        data_rank = data[start_idx:end_idx]
        completion_dict = []
        progress_bar = tqdm(range(0, len(data_rank), batch_size), disable=(rank != 0))
        for i in progress_bar:
            batch = data_rank[i : i + batch_size] if i + batch_size <= len(data_rank) else data_rank[i:]
            questions = [item['sent'] for item in batch]
            prompts = [get_conversation(q, processor) for q in questions]
            image_paths = []
            for item in batch:
                path_dir = item['img_id'].split('_')[1]
                image_path = f"/data1/TAP/data/mlm_pic/coco14_pic/{path_dir}/{item['img_id']}.jpg"
                image_paths.append(image_path)
            raw_images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = processor(
                    images=raw_images,
                    text=prompts,
                    return_tensors='pt',
                    padding=True
                ).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            results = processor.batch_decode(outputs, skip_special_tokens=True)
            for j, item in enumerate(batch):
                result_text = results[j].split("ASSISTANT:")[-1].strip()
                label = max(item['label'], key=item['label'].get) if item.get('label') else None
                completion_dict.append({
                    "question": questions[j],
                    "image_path": image_paths[j],
                    "answer": result_text,
                    "label": label
                })
        task_result_dict[test_data] = completion_dict
    
    # df = pd.read_parquet('/data2/TAP/data/flaviagiammarino/vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet', engine='pyarrow')
    # data = df.to_dict(orient='records')
    # num_samples = len(data)
    
    # samples_per_rank = num_samples // world_size
    # start_idx = rank * samples_per_rank
    # end_idx = start_idx + samples_per_rank
    # if rank == world_size - 1:
    #     end_idx = num_samples

    # data_rank = data[start_idx:end_idx]
    # completion_dict = []
    # progress_bar = tqdm(range(0, len(data_rank), batch_size), disable=(rank != 0))
    # for i in progress_bar:
    #     batch = data_rank[i : i + batch_size] if i + batch_size <= len(data_rank) else data_rank[i:]
    #     questions = [item['question'] for item in batch]
    #     prompts = [get_conversation(q, processor) for q in questions]
    #     image_list = []
    #     for item in batch:
    #         image_list.append(item['image']['bytes'])   
    #     raw_images = [Image.open(BytesIO(p)).convert("RGB") for p in image_list]

    #     inputs = processor(
    #             images=raw_images,
    #             text=prompts,
    #             return_tensors='pt',
    #             padding=True
    #         ).to(model.device)
    #     outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    #     results = processor.batch_decode(outputs, skip_special_tokens=True)
    #     for j, item in enumerate(batch):
    #         result_text = results[j].split("ASSISTANT:")[-1].strip()
    #         label = item['answer']
    #         completion_dict.append({
    #             "question": questions[j],
    #             # "image_bytes": image_list[j],
    #             "answer": result_text,
    #             "label": label
    #         })
    # task_result_dict['vqa_rad_raw'] = completion_dict
    
    # df = pd.read_parquet('/data2/TAP/data/HuggingFaceM4/ChartQA/data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet', engine='pyarrow')
    # df = df[df['human_or_machine'] == 0]
    # data = df.to_dict(orient='records')
    # num_samples = len(data)
    
    # samples_per_rank = num_samples // world_size
    # start_idx = rank * samples_per_rank
    # end_idx = start_idx + samples_per_rank
    # if rank == world_size - 1:
    #     end_idx = num_samples

    # data_rank = data[start_idx:end_idx]
    # completion_dict = []
    # progress_bar = tqdm(range(0, len(data_rank), batch_size), disable=(rank != 0))
    # for i in progress_bar:
    #     batch = data_rank[i : i + batch_size] if i + batch_size <= len(data_rank) else data_rank[i:]
    #     questions = [item['query'] for item in batch]
    #     prompts = [get_conversation(q, processor) for q in questions]
    #     image_list = []
    #     for item in batch:
    #         image_list.append(item['image']['bytes'])   
    #     raw_images = [Image.open(BytesIO(p)).convert("RGB") for p in image_list]

    #     inputs = processor(
    #             images=raw_images,
    #             text=prompts,
    #             return_tensors='pt',
    #             padding=True
    #         ).to(model.device)
    #     outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    #     results = processor.batch_decode(outputs, skip_special_tokens=True)
    #     for j, item in enumerate(batch):
    #         result_text = results[j].split("ASSISTANT:")[-1].strip()
    #         label = item['label'][0]
    #         completion_dict.append({
    #             "question": questions[j],
    #             # "image_bytes": image_list[j],
    #             "answer": result_text,
    #             "label": label
    #         })
    # task_result_dict['ChartQA'] = completion_dict

    data = read_json_or_jsonl("/data2/TAP/data/nlphuji/flickr30k/data/test.json")
    num_samples = len(data)
    
    samples_per_rank = num_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    if rank == world_size - 1:
        end_idx = num_samples

    data_rank = data[start_idx:end_idx]
    completion_dict = []
    progress_bar = tqdm(range(0, len(data_rank), batch_size), disable=(rank != 0))
    for i in progress_bar:
        batch = data_rank[i : i + batch_size] if i + batch_size <= len(data_rank) else data_rank[i:]
        questions = [None for item in batch]
        prompts = [get_conversation(q, processor) for q in questions]
        image_list = []
        for item in batch:
            image_list.append(f"/data2/TAP/data/nlphuji/flickr30k/flickr30k-images/{item['filename']}")   
        raw_images = [Image.open(p).convert("RGB") for p in image_list]

        inputs = processor(
                images=raw_images,
                text=prompts,
                return_tensors='pt',
                padding=True
            ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        results = processor.batch_decode(outputs, skip_special_tokens=True)
        for j, item in enumerate(batch):
            result_text = results[j].split("ASSISTANT:")[-1].strip()
            label = item['raw']
            completion_dict.append({
                # "image_bytes": image_list[j],
                "answer": result_text,
                "label": label
            })
    task_result_dict['flickr30k'] = completion_dict
   
    return task_result_dict