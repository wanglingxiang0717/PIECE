import argparse
import os
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from collections import OrderedDict
from torch.utils.data import DataLoader, SequentialSampler
from datetime import datetime
from typing import Callable, Any

from .extract_and_save import generation_mask_file, generation_mask_file_layer, process_topk_parameters

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def log_info(message: str, rank: int):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0(f"[{now}][PIECE info]: {message}", rank=rank)

def get_grad_avg(file_path):
    grad_all = 0.0
    shape_all = 0.0
    for file_name in os.listdir(file_path):
        if file_name.endswith('.pt'):
            file_grad_abs = os.path.join(file_path, file_name)
            grad_norm = torch.load(file_grad_abs)
            grad_norm = grad_norm.abs()
            grad_all += grad_norm.sum().item()
            shape_all += len(grad_norm.reshape(-1))
    return grad_all / shape_all

def save_file(file_path, file_save_path):
    grad_avg = get_grad_avg(file_path)
    for file_name in tqdm(os.listdir(file_path), desc="Processing modules"):
        if file_name.endswith('.pt'):
            file_grad_abs = os.path.join(file_path, file_name)
            file_grad_2 = os.path.join(file_path.replace("param_grad", "param_grad_2"), file_name)
            
            grad_norm = torch.load(file_grad_abs)
            grad_norm = grad_norm.abs()
            grad_norm_2 = torch.load(file_grad_2)
            grad_norm_save = grad_norm.reshape(-1) 
            grad_norm_save_2 = grad_norm_2.reshape(-1)
            grad_norm_save_new_2 = (grad_norm_save / torch.sqrt(grad_norm_save_2 + 1)).reshape(grad_norm.shape)

            file_save_new_2 = os.path.join(file_save_path, file_name)

            torch.save(grad_norm_save_new_2, file_save_new_2)

def generation_file(args, file_path):
    file_save_path = file_path.replace("param_grad", f"param_S")
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)
    save_file(file_path, file_save_path)

def standard_loss_function(model, batch, device, args):
    if 'sources' in batch:
        del batch['sources']
    batch = to_device(batch, device)
    loss = model(**batch, use_cache=False).loss
    return loss

def collect_local_grads(model, train_dataloader, device, args, max_steps=None, cpu_offload=False,
                        loss_function=standard_loss_function):
    total_steps = len(train_dataloader)
    progress_bar = tqdm(total=total_steps, leave=True, disable=(args.local_rank != 0))

    model_wrapped = model.module if hasattr(model, "module") else model

    local_sum = OrderedDict()
    local_sum2 = OrderedDict()

    for step, batch in enumerate(train_dataloader):
        # if 'sources' in batch:
        #     del batch['sources']

        # batch = to_device(batch, device)
        # loss = model(**batch, use_cache=False).loss
        loss = loss_function(model, batch, device, args)
        
        if args.local_rank == 0:
            progress_bar.update(1)
            description = f"Step {step}, Loss: {loss.item():.4f}"
            progress_bar.set_description(description, refresh=False)

        loss.backward()
        for name, p in model_wrapped.named_parameters():
            if p.grad is None:
                continue

            g = torch.nan_to_num(p.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            if cpu_offload:
                g_2 = g ** 2
                g = g.to("cpu", non_blocking=True)
                g_2 = g_2.to("cpu", non_blocking=True)
                if name not in local_sum:
                    local_sum[name] = g
                    local_sum2[name] = g_2
                else:
                    local_sum[name].add_(g)
                    local_sum2[name].add_(g_2)
            else:     
                if name not in local_sum:
                    local_sum[name] = g.clone()
                    local_sum2[name] = (g ** 2).clone()
                else:
                    local_sum[name].add_(g)
                    local_sum2[name].add_(g ** 2)
                    
        model.zero_grad()

        if max_steps is not None and step + 1 >= max_steps:
            break

    return local_sum, local_sum2

def allreduce_grad_dicts(local_sum, local_sum2):
    if not (dist.is_available() and dist.is_initialized()):
        return local_sum, local_sum2
    agg_sum = OrderedDict()
    agg_sum2 = OrderedDict()
    for name in local_sum.keys():
        t1 = local_sum[name]
        t2 = local_sum2[name]
        assert t1.is_cuda and t2.is_cuda

        dist.all_reduce(t1, op=dist.ReduceOp.SUM)
        dist.all_reduce(t2, op=dist.ReduceOp.SUM)

        agg_sum[name] = t1
        agg_sum2[name] = t2
    return agg_sum, agg_sum2

def save_grad_epoch(save_dir, grad_dict, grad_dict_2):
    save_dir_path = f"{save_dir}/param_grad/pt"
    save_dir_path_2 = f"{save_dir}/param_grad_2/pt"
    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(save_dir_path_2, exist_ok=True)

    for name, grad_sq in grad_dict.items():
        filename = f"{save_dir_path}/{name}.pt"
        torch.save(grad_sq.cpu(), filename)
    grad_dict.clear()

    for name, grad_sq_2 in grad_dict_2.items():
        filename = f"{save_dir_path_2}/{name}.pt"
        torch.save(grad_sq_2.cpu(), filename)
    grad_dict_2.clear()

def process_and_save_mask_multiGPU(model, dataloader, save_dir, args, loss_function=standard_loss_function):
    log_info("masking_process_start", args.local_rank)
    if dist.is_available() and dist.is_initialized():
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)  
    else:
        device = torch.device("cuda")
    local_sum, local_sum2 = collect_local_grads(model, dataloader, device, args, loss_function=loss_function)
    agg_sum, agg_sum2 = allreduce_grad_dicts(local_sum, local_sum2)
    save_grad_epoch(save_dir, agg_sum, agg_sum2)
    
def process_and_save_mask_singleGPU(model, dataset, data_collator, save_dir, args, cpu_offload=False, loss_function=standard_loss_function):
    log_info("Mask process start", args.local_rank)
    if dist.is_available() and dist.is_initialized():
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)  
    else:
        device = torch.device("cuda")
    if args.local_rank == 0 or args.local_rank == -1:
        data_sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                collate_fn=data_collator,
                                sampler=data_sampler,
                                batch_size=1)
        local_sum, local_sum2 = collect_local_grads(model, dataloader, device, args, cpu_offload=cpu_offload, loss_function=loss_function)
        save_grad_epoch(save_dir, local_sum, local_sum2)

def capture_grad_hook(name, file_path=None):
    cached_mask = None  

    def hook_fn(grad):
        nonlocal cached_mask
        if cached_mask is None:  
            mask = torch.zeros_like(grad, device=grad.device)
            if file_path:
                json_file = f"{file_path}/{name}.json"
                if os.path.exists(json_file):
                    with open(json_file, 'r') as f_r:
                        data = json.load(f_r)
                        for pos in data["positions"]:
                            try:
                                mask[tuple(pos)] = 1.0
                            except IndexError:
                                # print(f"[警告] 索引 {pos} 超出梯度形状 {grad.shape}，跳过")
                                print(f"[WARNING] Index {pos} exceeds gradient shape {grad.shape}, skipping")
                else:
                    mask.fill_(1.0)  
            else:
                mask.fill_(1.0)  

            cached_mask = mask 
        else:
            mask = cached_mask

        return grad * mask

    return hook_fn

# def process_mask(model, dataset, data_collator, save_dir, mode, top_ratio, args, singleGPU=True):
def process_mask(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        data_collator,
        save_dir: str,
        mode: str,
        top_ratio: float,
        args: argparse.Namespace,
        singleGPU: bool = True,
        cpu_offload: bool = False,
        loss_function: Callable[
            [torch.nn.Module, Any, torch.device, argparse.Namespace], 
            torch.Tensor
        ] = standard_loss_function,
    ):
    """
    Process and save parameter masks for the model.
        model (torch.nn.Module): The model whose parameters are to be masked.
        dataset (torch.utils.data.Dataset): The dataset used for gradient computation.
        data_collator: Data collator for batching (e.g., DataCollatorForSeq2Seq).
        save_dir (str): Directory to save intermediate and final mask files.
        mode (str): Mode of masking, either 'F' (Fisher) or 'S' (Second-order normalization).
        top_ratio (float): Fraction of top parameters to keep (e.g., 0.001 for top 0.1%).
        args (argparse.Namespace): Arguments containing configuration like local_rank.
        singleGPU (bool, optional): Whether to perform gradient computation on a single GPU.
            Defaults to True. Recommended for safety and consistency.
        cpu_offload (bool, optional): Whether to offload gradient computation to CPU to save GPU memory.
            Defaults to False. (more compute time cost)
        loss_function : Callable, optional (default = `standard_loss_function`)
            A user-definable loss function with signature:
                loss = loss_function(model, batch, device, args)
            This allows custom training objectives or model forward passes.
    """
    assert mode in ["F", "S"], "Currently only 'F' (Fisher) and 'S' (Second-order normalization) masking modes are supported."
    assert singleGPU == True, "Gradient computation is currently recommended on a single GPU."
    process_and_save_mask_singleGPU(model, dataset, data_collator, save_dir, args, cpu_offload=cpu_offload, loss_function=loss_function)
    if args.local_rank == 0 or args.local_rank == -1:
        if mode == "S":
            generation_file(args, file_path=f"{save_dir}/param_grad/pt")
            file_detection_path = f"{save_dir}/param_S/pt"
        else:
            file_detection_path = f"{save_dir}/param_grad_2/pt"
        mask_save_dir = file_detection_path.replace('pt', f'top{top_ratio}')
        if not os.path.exists(mask_save_dir):
            process_topk_parameters(file_detection_path, mask_save_dir, top_ratio=float(top_ratio), device="cpu")
    log_info("Mask process finish", args.local_rank)

# def mask_grads(model, mode, save_dir, top_ratio, mask_file_path=None):
def mask_grads(
        model: torch.nn.Module,
        mode: str,
        save_dir: str,
        top_ratio: float,
        mask_file_path: str = None
    ):
    """
    Register gradient hooks on the model's parameters to selectively mask gradients 
    for later processing or saving.

    Args:
        model (torch.nn.Module): The model whose parameters will have gradient hooks registered.
        mode (str): Masking mode, either 'F' (Fisher) or 'S' (Second-order normalization).
        save_dir (str): Directory where intermediate and final mask files are stored.
        top_ratio (float): Fraction of top parameters to keep (e.g., 0.001 for top 0.1%).
        mask_file_path (str, optional): Custom path to save mask information. 
            If None, defaults to:
                - "{save_dir}/param_S/top{top_ratio}" for mode 'S'
                - "{save_dir}/param_grad_2/top{top_ratio}" otherwise

    Notes:
        - Hooks are only registered on parameters with `requires_grad=True`.
        - For distributed models wrapped in `torch.nn.DataParallel` or `DistributedDataParallel`,
          the underlying module is used (`model.module`) for hook registration.
    """
    if mask_file_path is None and mode == "S":
        mask_file_path = f"{save_dir}/param_S/top{top_ratio}"
    elif mask_file_path is None:
        mask_file_path = f"{save_dir}/param_grad_2/top{top_ratio}"
    
    model_for_hook = model.module if hasattr(model, "module") else model    
    for name, p in model_for_hook.named_parameters():
        if not p.requires_grad:
            continue
        p.register_hook(capture_grad_hook(name, mask_file_path))
