from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
import transformers
import torch
import csv
import os
import pandas as pd
from tqdm import tqdm
import jsonlines
import torch.distributed as dist
# from test_humaneval_one import get_result

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def generate_one_completion(prompt, model, tokenizer):
    # 确保模型处于评估模式
    model.eval()
    
    sequences = model.generate(
        input_ids=tokenizer(prompt, return_tensors='pt').input_ids.to(model.device),
        do_sample=True,
        top_k=10,
        temperature=0.3,
        top_p=0.95,
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    return generated_text.replace(prompt, "")

def process_get_result(base_model, device_id):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": int(device_id)},
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
            "text-generation", #指定任务
            model=model,
            tokenizer = tokenizer,
            torch_dtype=torch.float16,
            # device_map={"": 0}, # 运行设备
        )

    problems = read_problems()
    num_samples_per_task = 1
    samples = [
        dict(task_id=task_id, completion=filter_code(generate_one_completion(problems[task_id]["prompt"], model, tokenizer)))
        for task_id in tqdm(problems)
        for _ in range(num_samples_per_task)
    ]
    write_jsonl(f"{base_model}/humaneval_dosample_test.jsonl", samples)
    # del tokenizer
    # get_result(f"{base_model}/humaneval_dosample_test.jsonl")    

def process_get_result_file(model, tokenizer):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    problems = read_problems()
    num_samples_per_task = 1
    problem_ids = list(problems.keys())
    total = len(problem_ids)
    per_rank = total // world_size
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank
    if rank == world_size - 1:
        end_idx = total
    local_problem_ids = problem_ids[start_idx:end_idx]
    # samples = [
    #     dict(task_id=task_id, completion=filter_code(generate_one_completion(problems[task_id]["prompt"], model, tokenizer)))
    #     for task_id in tqdm(problems)
    #     for _ in range(num_samples_per_task)
    # ]
    local_samples = []
    for task_id in tqdm(local_problem_ids, disable=(rank != 0)):
        prompt = problems[task_id]["prompt"]
        completion = filter_code(generate_one_completion(prompt, model, tokenizer))
        local_samples.append(dict(task_id=task_id, completion=completion))
    return local_samples

    