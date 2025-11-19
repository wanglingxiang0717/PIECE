import sys
sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
from tqdm import tqdm
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import shutil
import random
import subprocess
import torch
from param_protection.get_parameters_import_files import generation_file
from param_protection.extract_and_save_new import generation_mask_file, generation_mask_file_layer
mask_list = {
    "Fisher": "parameters_grad_2",
    "ours": "parameters_import_new_2"
}


class GPUOccupier:
    def __init__(self, device_id: int, memory_gb: int):
        self.device_id = device_id
        self.memory_gb = memory_gb
        self.tensor = None

    def occupy(self):
        torch.cuda.set_device(self.device_id)
        num_elements = self.memory_gb * 1024**3 // 4  # float32 4 bytes
        print(f"[ENTER] 尝试在 GPU:{self.device_id} 占用 {self.memory_gb} GB 显存...")
        try:
            self.tensor = torch.empty(num_elements, dtype=torch.float32, device=f"cuda:{self.device_id}")
            print(f"[ENTER] GPU:{self.device_id} 已成功占用 {self.memory_gb} GB 显存")
        except RuntimeError as e:
            print(f"[ERROR] GPU:{self.device_id} 分配失败: {e}")

    def release(self):
        if self.tensor is not None:
            del self.tensor
            torch.cuda.empty_cache()
            print(f"[EXIT] GPU:{self.device_id} 显存已释放")


class MultiGPUOccupier:
    def __init__(self, gpu_mem_map: dict[int, int]):
        self.gpu_mem_map = gpu_mem_map
        self.occupiers: list[GPUOccupier] = []

    def __enter__(self):
        self.occupiers = [GPUOccupier(gpu, mem) for gpu, mem in self.gpu_mem_map.items()]
        for occ in self.occupiers:
            occ.occupy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for occ in self.occupiers:
            occ.release()

def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=64,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=32,
        help="The maximum sequence length.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    
    parser.add_argument('--mask_method',
            default=None,
            help='grad mask file method')

    parser.add_argument("--target_name",
                    type=str,
                    default=None,
                    help="The Name of datasets.")
    
    parser.add_argument("--top_ratio",
                type=float,
                default=0.0001,
                help="The top_ratio selection.")
    args = parser.parse_args()

    return args

def run_training(args):
    output_model = f"/data1/TAP/model_exp_mllm/1020_{args.target_name}_{args.mask_method}_parameters_test_epoch1_random_1000"
    os.makedirs(output_model, exist_ok=True)
    os.makedirs(f"{output_model}/log/", exist_ok=True)
    port = random.randint(25000, 30000)
    # if args.target_name in ["MeetingBank", "ScienceQA", "20Minuten", "Py150"]:
    #     batch_size = str(4)
    # else:
    #     batch_size = str(8)
    batch_size = str(4)
    cmd = [
        "python", "/data2/TAP/code/PIECE_mllm_qwen/training_parameters_try/main.py",
        "--data_path", f"{args.data_path}",
        "--model_name_or_path", f"{args.model_name_or_path}",
        "--per_device_train_batch_size", batch_size,
        "--per_device_eval_batch_size", "1",
        "--max_prompt_len", f"{args.max_prompt_len}",
        "--max_ans_len", f"{args.max_ans_len}",
        "--learning_rate", "1e-5",
        "--weight_decay", "0.0",
        "--num_train_epochs", "1",
        "--gradient_accumulation_steps", "1",
        "--lr_scheduler_type", "constant",
        "--num_warmup_steps", "0",
        "--seed", f"{args.seed}",
        "--zero_stage", f"2",
        "--deepspeed",
        "--print_loss",
        "--CL_method", f"base",
        "--enable_tensorboard",
        "--tensorboard_path", f"{output_model}/log/",
        "--offload",
        "--output_dir", output_model,
        "--param_import_savepath", f"{output_model}/parameters_grad"
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["MASTER_PORT"] = str(port)

    log_file = os.path.join(output_model, "train.log")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")   # 打印到终端
            f.write(line)         # 写入日志
        process.wait()
    return process.returncode

def get_grad_mask(model, train_dataloader, device):
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

    for step, batch in enumerate(train_dataloader):
        if 'sources' in batch:
            del batch['sources']
        batch = to_device(batch, device)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss

        # if self.args.global_rank == 0:
        #     progress_bar.update(1)
        #     description = f"Step {step}, Loss: {loss.item():.4f}"
        #     progress_bar.set_description(description, refresh=False)

        loss.backward()
        model.zero_grad()

    for h in hook_handles:
        h.remove()
    hook_handles.clear()

    return grad_dict, grad_dict_2

def process_mask_path(args):
    file_path = f"/data1/TAP/model_exp_mllm/1020_{args.target_name}_{args.mask_method}_parameters_test_epoch1_random_1000/parameters_grad/epoch0"
    if not os.path.exists(file_path):
        with MultiGPUOccupier({0: 70, 2: 70, 3: 70}):
            result = run_training(args)
            if result == 0:
                    print("Training finished successfully.")
            else:
                print(f"Training failed with exit code {result.returncode}")
        with MultiGPUOccupier({0: 70, 1: 70, 2: 70, 3: 70}):
            generation_file(args, file_path=f"/data1/TAP/model_exp_mllm/1020_{args.target_name}_{args.mask_method}_parameters_test_epoch1_random_1000/parameters_grad/epoch0")
    
    file_dection_path = f"/data1/TAP/model_exp_mllm/1020_{args.target_name}_{args.mask_method}_parameters_test_epoch1_random_1000/{args.last_name}/epoch0"
    save_dir = file_dection_path.replace("epoch0", f"top{args.top_ratio}")
    if not os.path.exists(save_dir):
        with MultiGPUOccupier({0: 70, 1: 70, 2: 70, 3: 70}):
            generation_mask_file(args, file_dection_path)

if __name__ == "__main__":
    args = parse_args()
    args.last_name = mask_list[args.mask_method]
    # stop = input(args.last_name)
    process_mask_path(args)