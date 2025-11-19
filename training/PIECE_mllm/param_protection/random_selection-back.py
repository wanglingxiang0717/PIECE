# from transformers import AutoModelForCausalLM, AutoTokenizer
# # from transformers import LlamaForCausalLM, LlamaConfig
# model_path = "/data/wanglingxiang/model/Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# for name, param in model.named_parameters():
#     print(name)
#     stop = input(param.shape)

import os
import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

def save_random_selection(seed=42):
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_path = "/data/wanglingxiang/model/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    save_dir = f"/home/wanglingxiang/random_001_parameters_{seed}"
    os.makedirs(save_dir, exist_ok=True)

    sample_ratio = 0.0001  # 0.01%

    param_infos = [] 

    flat_index_start = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param_infos.append((name, param.shape, flat_index_start, flat_index_start + numel))
        flat_index_start += numel

    total_elements = flat_index_start
    num_samples = max(1, int(total_elements * sample_ratio))
    print(f"Total params: {total_elements} | Sampling {num_samples} elements")

    sampled_flat_indices = set(random.sample(range(total_elements), num_samples))

    for name, shape, flat_start, flat_end in tqdm(param_infos):
        local_indices = [idx - flat_start for idx in sampled_flat_indices if flat_start <= idx < flat_end]
        if not local_indices:
            continue

        multi_indices = [list(map(int, np.unravel_index(idx, shape))) for idx in local_indices]

        filename = name + ".json"
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "w") as f:
            json.dump({"positions": multi_indices}, f)

        # print(f"Saved {len(multi_indices)} positions for {name} to {save_path}")

if __name__ == "__main__":
    for seed in range(40, 50):
        save_random_selection(seed)