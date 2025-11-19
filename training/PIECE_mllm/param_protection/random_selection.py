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

import os
import random
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def save_random_selection(seed=42, target=None, end_name=None):
    random_seed = seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_path = "/data1/TAP/model/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    save_dir = f"/data2/TAP/data/random_param_selection_llama3_8B_ins/random_001_parameters_{seed}_{end_name}"
    os.makedirs(save_dir, exist_ok=True)

    sample_ratio = 0.0001  # 0.01%

    total_elements = 0
    filtered_param_infos = []
    filtered_total_elements = 0

    # 遍历模型所有参数，记录总元素数，同时收集符合 target 的参数
    for name, param in model.named_parameters():
        numel = param.numel()
        total_elements += numel

        if target is not None and isinstance(target, list):
            if any(key in name for key in target):
                filtered_param_infos.append((name, param.shape, filtered_total_elements, filtered_total_elements + numel))
                filtered_total_elements += numel
        else:
            filtered_param_infos.append((name, param.shape, filtered_total_elements, filtered_total_elements + numel))
            filtered_total_elements += numel

    # 采样数量：基于全模型大小
    num_samples = max(1, int(total_elements * sample_ratio))
    print(f"Total elements in model: {total_elements}")
    print(f"Filtered elements: {filtered_total_elements}")
    print(f"Sampling {num_samples} elements (based on total), from filtered params only.")

    # 从 filtered 空间中采样
    if filtered_total_elements <= num_samples:
        sampled_flat_indices = set(range(filtered_total_elements))
        print(f"Filtered elements ({filtered_total_elements}) <= num_samples ({num_samples}), selecting all.")
    else:
        sampled_flat_indices = set(random.sample(range(filtered_total_elements), num_samples))

    # 匹配 sampled_flat_indices 到各个参数张量
    for name, shape, flat_start, flat_end in tqdm(filtered_param_infos):
        local_indices = [idx - flat_start for idx in sampled_flat_indices if flat_start <= idx < flat_end]

        if not local_indices:
            continue

        multi_indices = [list(map(int, np.unravel_index(idx, shape))) for idx in local_indices]

        save_path = os.path.join(save_dir, name + ".json")
        with open(save_path, "w") as f:
            json.dump({"positions": multi_indices}, f)

if __name__ == "__main__":
    target_selection_pool = [
            "input_layernorm",
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "model.norm",
            "lm_head"
        ]
    target = None
    target = [target_selection_pool[0], target_selection_pool[5], target_selection_pool[-2]]
    # target = target_selection_pool[6:9]
    stop = input(target)
    name = "norm"
    for seed in range(40, 43):
        save_random_selection(seed, target, name)