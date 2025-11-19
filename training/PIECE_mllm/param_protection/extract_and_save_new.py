import os
import torch
import json
from tqdm import tqdm
from collections import defaultdict
import time
import numpy as np

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
