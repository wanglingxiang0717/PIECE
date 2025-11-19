import argparse
import os
import math
from tqdm import tqdm
import shutil
import random
import subprocess
import torch
from collections import defaultdict
import time
import numpy as np
import torch.distributed as dist
import json
from utils.utils import print_rank_0, to_device, save_hf_format, save_hf_format_task, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer

# from param_protection.get_parameters_import_files import generation_file
# from param_protection.extract_and_save_new import generation_mask_file, generation_mask_file_layer
mask_list = {
    "Fisher": "parameters_grad_2",
    "ours": "parameters_import_new_2"
}
def get_all_parameters(grad_dir, device="cpu"):
    start_time = time.time()
    files = [f for f in os.listdir(grad_dir) if f.endswith(".pt")]
    all_tensors = []
    layer_shapes = {}
    layer_offsets = {}
    offset = 0

    for f in files:
        layer_name = f[:-3]
        t = torch.load(os.path.join(grad_dir, f), map_location=device).float()
        if "embed" in layer_name.lower():
            t.zero_()
        layer_shapes[layer_name] = t.shape
        layer_offsets[layer_name] = offset
        flat_t = t.reshape(-1)
        all_tensors.append(flat_t)
        offset += flat_t.numel()

    merged = torch.cat(all_tensors, dim=0)
    end_time = time.time()
    print(f"[读取参数] 耗时 {end_time - start_time:.2f}s, 总参数量: {merged.numel()}")
    return merged, layer_shapes, layer_offsets, files

def topk_threshold_indices(merged, k):
    start_time = time.time()
    topk_values, topk_indices = torch.topk(merged, k)
    threshold = topk_values.min().item()
    end_time = time.time()
    print(f"[计算 top-k] 耗时 {end_time - start_time:.2f}s, 阈值: {threshold}")
    return topk_indices, threshold

def unravel_indices(indices, layer_shapes, layer_offsets):
    indices_np = indices.cpu().numpy()
    layer_names, starts, ends, shapes = [], [], [], []

    for name, offset in layer_offsets.items():
        numel = np.prod(layer_shapes[name])
        layer_names.append(name)
        starts.append(offset)
        ends.append(offset + numel)
        shapes.append(layer_shapes[name])

    starts = np.array(starts)
    ends = np.array(ends)

    idx_layer = np.searchsorted(ends, indices_np, side='right')

    layer_positions = defaultdict(list)
    for i in tqdm(range(len(indices_np)), desc="还原索引"):
        layer_idx = idx_layer[i]
        layer_name = layer_names[layer_idx]
        local_idx = indices_np[i] - starts[layer_idx]
        pos = np.unravel_index(local_idx, shapes[layer_idx])
        layer_positions[layer_name].append(list(pos))

    return layer_positions


def save_layer_positions(layer_positions, save_dir):
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)

    for layer, items in tqdm(layer_positions.items(), desc="保存 JSON"):
        # 将 numpy.int64 或其他 numpy 类型统一转换为 Python int
        items_py = [[int(x) for x in pos] for pos in items]

        output_path = os.path.join(save_dir, f"{layer}.json")
        with open(output_path, "w") as f:
            json.dump({"positions": items_py}, f)

    end_time = time.time()
    print(f"[保存 JSON] 耗时 {end_time - start_time:.2f}s")

def process_topk_parameters(grad_dir, save_dir, top_ratio=0.0001, device="cpu"):
    total_start = time.time()
    
    merged, layer_shapes, layer_offsets, files = get_all_parameters(grad_dir, device=device)
    merged = merged.clone()
    # merged[torch.isnan(merged)] = 0.0
    # merged[torch.isinf(merged)] = 0.0
    merged = torch.nan_to_num(merged, nan=0.0, posinf=float("inf"), neginf=-float("inf"))
    total_params = merged.numel()
    k = max(1, int(total_params * top_ratio))
    # k = 803026
    topk_indices, threshold = topk_threshold_indices(merged, k)
    layer_positions = unravel_indices(topk_indices, layer_shapes, layer_offsets)
    save_layer_positions(layer_positions, save_dir)

    total_end = time.time()
    print(f"\n[总耗时] {total_end - total_start:.2f}s, top-{top_ratio*100}% 参数数量: {k}, 阈值: {threshold}")

def process_topk_parameters_per_layer(grad_dir, save_dir, top_ratio=0.001, device="cpu"):
    total_start = time.time()
    os.makedirs(save_dir, exist_ok=True)

    files = [f for f in os.listdir(grad_dir) if f.endswith(".pt")]

    grouped_files = defaultdict(list)
    for f in files:
        parts = f.split(".")
        if parts[0] == "model" and parts[1] == "layers":
            layer_id = ".".join(parts[:3])  # "model.layers.0"
        else:
            layer_id = parts[0]  # fallback
        grouped_files[layer_id].append(f)

    for layer_name, layer_files in tqdm(grouped_files.items(), desc="逐层处理"):
        module_shapes = {}
        module_offsets = {}
        offset = 0
        flat_tensors = []

        for f in sorted(layer_files):
            module_name = f  # 保留完整文件名作为 key
            t = torch.load(os.path.join(grad_dir, f), map_location=device).float()
            if "embed" in module_name.lower():
                t.zero_()
            t = torch.nan_to_num(t, nan=0.0, posinf=float("inf"), neginf=-float("inf"))

            module_shapes[module_name] = t.shape
            module_offsets[module_name] = offset

            flat_t = t.reshape(-1)
            flat_tensors.append(flat_t)
            offset += flat_t.numel()

        merged = torch.cat(flat_tensors, dim=0)

        numel = merged.numel()
        k = max(1, int(numel * top_ratio))
        topk_values, topk_indices = torch.topk(merged, k)
        threshold = topk_values.min().item()

        module_positions = defaultdict(list)
        starts = {m: module_offsets[m] for m in module_offsets}
        ends = {m: module_offsets[m] + np.prod(module_shapes[m]) for m in module_shapes}

        for idx in topk_indices.cpu().numpy():
            for m in module_shapes:
                if starts[m] <= idx < ends[m]:
                    local_idx = idx - starts[m]
                    pos = np.unravel_index(local_idx, module_shapes[m])
                    module_positions[m].append([int(x) for x in pos])
                    break

        for m, pos in module_positions.items():
            output_path = os.path.join(save_dir, f"{m.replace('.pt', '')}.json")
            with open(output_path, "w") as fjson:
                json.dump({"threshold": threshold, "positions": pos}, fjson)

    total_end = time.time()
    print(f"\n[逐层(分组)模式完成] 总耗时 {total_end - total_start:.2f}s, 每层 top-{top_ratio*100:.4f}% 已保存到 {save_dir}")
  
def generation_mask_file(args, grad_dir):
    save_dir = grad_dir.replace("epoch0", f"top{args.top_ratio}")
    process_topk_parameters(grad_dir, save_dir, top_ratio=float(args.top_ratio), device="cpu")
    # return str(save_dir)
    
def generation_mask_file_layer(args, grad_dir):
    save_dir = grad_dir.replace("epoch0", f"top{args.top_ratio}_layer")
    process_topk_parameters_per_layer(grad_dir, save_dir, top_ratio=float(args.top_ratio), device="cpu")
    # return str(save_dir)

def generation_mask_file_simple(top_ratio, grad_dir):
    save_dir = grad_dir.replace("epoch0", f"top{top_ratio}")
    process_topk_parameters(grad_dir, save_dir, top_ratio=float(top_ratio), device="cpu")

def get_grad_mask(model, train_dataloader, device):
    model.train()
    grad_dict = {}
    grad_dict_2 = {}
    hook_handles = []

    def capture_grad_hook(name):
        def hook(grad):
            if grad is None:
                return
            grad_sq = grad.detach()
            grad_sq = torch.nan_to_num(grad_sq, nan=0.0)
            grad_sq_2 = grad_sq ** 2
            grad_sq = grad_sq.to("cpu", non_blocking=True)
            grad_sq_2 = grad_sq_2.to("cpu", non_blocking=True)
            if name in grad_dict:
                grad_dict[name].add_(grad_sq)
            else:
                grad_dict[name] = grad_sq
            if name in grad_dict_2:
                grad_dict_2[name].add_(grad_sq_2)
            else:
                grad_dict_2[name] = grad_sq_2
        return hook

    model_for_hook = model.module if hasattr(model, "module") else model
    for name, param in model_for_hook.named_parameters():
        if param.requires_grad:
            handle = param.register_hook(capture_grad_hook(name))
            hook_handles.append(handle)
            
    rank = dist.get_rank() if dist.is_initialized() else 0
    progress_bar = tqdm(total=len(train_dataloader), disable=(rank != 0), desc="Computing Grad Mask")
    
    for step, batch in enumerate(train_dataloader):
        if 'sources' in batch:
            del batch['sources']
        batch = to_device(batch, device)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss

        model.backward(loss)
        model.zero_grad()
        if rank == 0:
            progress_bar.update(1)
            progress_bar.set_postfix(loss=float(loss.item()))
    progress_bar.close()   
    for h in hook_handles:
        h.remove()
    hook_handles.clear()

    return grad_dict, grad_dict_2

# def generation_mask_files():
    

# def process_mask_path(args):
#     file_path = f"/data1/TAP/model_exp_2b/1020_{args.target_name}_{args.mask_method}_parameters_test_epoch1_random_1000/parameters_grad/epoch0"
#     if not os.path.exists(file_path):
#         with MultiGPUOccupier({0: 70, 2: 70, 3: 70}):
#             result = run_training(args)
#             if result == 0:
#                     print("Training finished successfully.")
#             else:
#                 print(f"Training failed with exit code {result.returncode}")
#         with MultiGPUOccupier({0: 70, 1: 70, 2: 70, 3: 70}):
#             generation_file(args, file_path=f"/data1/TAP/model_exp_2b/1020_{args.target_name}_{args.mask_method}_parameters_test_epoch1_random_1000/parameters_grad/epoch0")
    
#     file_dection_path = f"/data1/TAP/model_exp_2b/1020_{args.target_name}_{args.mask_method}_parameters_test_epoch1_random_1000/{args.last_name}/epoch0"
#     save_dir = file_dection_path.replace("epoch0", f"top{args.top_ratio}")
#     if not os.path.exists(save_dir):
#         with MultiGPUOccupier({0: 70, 1: 70, 2: 70, 3: 70}):
#             generation_mask_file(args, file_dection_path)

# if __name__ == "__main__":
#     args = parse_args()
#     args.last_name = mask_list[args.mask_method]
#     # stop = input(args.last_name)
#     process_mask_path(args)