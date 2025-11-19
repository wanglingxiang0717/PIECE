import torch
from utils.utils import print_rank_0, to_device, save_hf_format, save_hf_format_task, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import json
import os
import time
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from transformers import GenerationConfig
from model.base_model import CL_Base_Model
# from utils.process_mask import generation_mask
from utils.process_mask_file import get_grad_mask, generation_mask_file
import shutil
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)
    
def save_grad_epoch(grad_dict, grad_dict_2, save_dir="/data2/TAP/data/save_grad/humaneval/"):
    save_dir_path = f"{save_dir}/epoch0"
    save_dir_path_2 = f"{save_dir.replace('parameters_grad', 'parameters_grad_2')}/epoch0"
    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(save_dir_path_2, exist_ok=True)

    # 存储时再转 CPU
    for name, grad_sq in grad_dict.items():
        filename = f"{save_dir_path}/{name}.pt"
        torch.save(grad_sq.cpu(), filename)
    grad_dict.clear()

    for name, grad_sq_2 in grad_dict_2.items():
        filename = f"{save_dir_path_2}/{name}.pt"
        torch.save(grad_sq_2.cpu(), filename)
    grad_dict_2.clear()
    return save_dir_path_2

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
                                print(f"[警告] 索引 {pos} 超出梯度形状 {grad.shape}，跳过")
                else:
                    mask.fill_(1.0)  
            else:
                mask.fill_(1.0)  

            cached_mask = mask 
        else:
            mask = cached_mask

        return grad * mask

    return hook_fn

def capture_grad_hook_slow(name, file_path=None):
    def hook_fn(grad, file_path=file_path):
        mask = torch.zeros_like(grad)
        if file_path:
            # stop = input(file_path)
            file_path = f"{file_path}/{name}.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f_r:
                    data = json.load(f_r)
                    for pos in data["positions"]:
                        try:
                            mask[tuple(pos)] = 1.0
                        except IndexError:
                            print(f"[警告] 索引 {pos} 超出梯度形状 {grad.shape}，跳过")
        else:
            mask = torch.ones_like(grad)
        return grad * mask  #通过梯度掩码的方式对某些参数实现保护(冻结)
    
    return hook_fn

class SPTCF(CL_Base_Model):
    def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args, lambda_ewc=400):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args)
        self.args.top_ratio = self.args.mask_top_ratio

    # def process_mask(self, task, save_file_dir, device):
    #     print_rank_0(
    #         f"***** Task: {task} generation mask*****",
    #         self.args.global_rank)
    #     local_grad_dict, loac_grad_dict_2 = get_grad_mask(self.model, self.train_task_list[task], device)
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()
    #     if rank == 0:
    #         gathered_grad_dicts = [None for _ in range(world_size)]
    #         gathered_grad_dicts_2 = [None for _ in range(world_size)]
    #     else:
    #         gathered_grad_dicts = None
    #         gathered_grad_dicts_2 = None

    #     dist.gather_object(local_grad_dict, gathered_grad_dicts, dst=0)
    #     dist.gather_object(local_grad_dict_2, gathered_grad_dicts_2, dst=0)
    #     if rank == 0:
    #         merged_grad = {}
    #         merged_grad_2 = {}

    #         for d in gathered_grad_dicts:
    #             for name, g in d.items():
    #                 if name not in merged_grad:
    #                     merged_grad[name] = g.clone()
    #                 else:
    #                     merged_grad[name] += g

    #         for d in gathered_grad_dicts_2:
    #             for name, g in d.items():
    #                 if name not in merged_grad_2:
    #                     merged_grad_2[name] = g.clone()
    #                 else:
    #                     merged_grad_2[name] += g
    #         # save_file_dir = os.path.join(self.args.mask_save_file_dir, f"{task}_mask")
    #         save_grad_epoch(merged_grad, merged_grad_2, save_dir=save_file_dir)
    #     dist.barrier()
    #     # self.model.train()

    def process_mask(self, task, save_file_dir, device):
        print_rank_0(f"***** Task: {task} generation mask*****", self.args.global_rank)

        # 计算梯度 mask
        local_grad_dict, local_grad_dict_2 = get_grad_mask(
            self.model, self.train_task_list[task], device
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        rank_dir = os.path.join(save_file_dir, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)

        for name, tensor in local_grad_dict.items():
            torch.save(tensor.clone().cpu(), os.path.join(rank_dir, f"{name}.pt"))
        for name, tensor in local_grad_dict_2.items():
            torch.save(tensor.clone().cpu(), os.path.join(rank_dir, f"{name}_2.pt"))

        dist.barrier()

        if rank == 0:
            merged_grad = {}
            merged_grad_2 = {}

            all_ranks = [os.path.join(save_file_dir, f"rank{r}") for r in range(world_size)]

            param_files = [f for rdir in all_ranks for f in os.listdir(rdir) if not f.endswith("_2.pt")]
            param_files.sort()
            for f in tqdm(param_files, desc="Aggregating grad dicts"):
                name = f.replace(".pt", "")
                merged_tensor = None
                for r in range(world_size):
                    t = torch.load(os.path.join(save_file_dir, f"rank{r}", f))
                    if merged_tensor is None:
                        merged_tensor = t
                    else:
                        merged_tensor += t
                merged_grad[name] = merged_tensor

            param_files_2 = [f for rdir in all_ranks for f in os.listdir(rdir) if f.endswith("_2.pt")]
            param_files_2.sort()
            for f in tqdm(param_files_2, desc="Aggregating grad dicts 2"):
                name = f.replace("_2.pt", "")
                merged_tensor = None
                for r in range(world_size):
                    t = torch.load(os.path.join(save_file_dir, f"rank{r}", f))
                    if merged_tensor is None:
                        merged_tensor = t
                    else:
                        merged_tensor += t
                merged_grad_2[name] = merged_tensor

            return_path = save_grad_epoch(merged_grad, merged_grad_2, save_dir=save_file_dir)
            for rdir in all_ranks:
                shutil.rmtree(rdir)

        dist.barrier()
        if rank == 0:
            return return_path    

    # def process_mask(self, task, save_file_dir, device):
    #     print_rank_0(f"***** Task: {task} generation mask*****", self.args.global_rank)

    #     # 计算梯度 mask
    #     local_grad_dict, local_grad_dict_2 = get_grad_mask(
    #         self.model, self.train_task_list[task], device
    #     )
    #     rank = dist.get_rank()
    #     world_size = dist.get_world_size()

    #     # 每个 rank 单独保存参数 tensor 文件
    #     rank_dir = os.path.join(save_file_dir, f"rank{rank}")
    #     os.makedirs(rank_dir, exist_ok=True)

    #     for name, tensor in local_grad_dict.items():
    #         torch.save(tensor.clone().cpu(), os.path.join(rank_dir, f"{name}.pt"))
    #     for name, tensor in local_grad_dict_2.items():
    #         torch.save(tensor.clone().cpu(), os.path.join(rank_dir, f"{name}_2.pt"))

    #     # 等待所有 rank 完成写入
    #     dist.barrier()

    #     # rank0 汇总
    #     if rank == 0:
    #         merged_grad = {}
    #         merged_grad_2 = {}

    #         # 进度条显示
    #         all_ranks = [os.path.join(save_file_dir, f"rank{r}") for r in range(world_size)]
    #         param_files = [f for rdir in all_ranks for f in os.listdir(rdir) if not f.endswith("_2.pt")]
    #         param_files.sort()  # 保证顺序一致

    #         for f in tqdm(param_files, desc="Aggregating grad dicts"):
    #             name = f.replace(".pt", "")
    #             merged_tensor = None
    #             for r in range(world_size):
    #                 t = torch.load(os.path.join(save_file_dir, f"rank{r}", f))
    #                 if merged_tensor is None:
    #                     merged_tensor = t
    #                 else:
    #                     merged_tensor += t
    #             merged_grad[name] = merged_tensor

    #         # grad_dict_2
    #         param_files_2 = [f for rdir in all_ranks for f in os.listdir(rdir) if f.endswith("_2.pt")]
    #         param_files_2.sort()
    #         for f in tqdm(param_files_2, desc="Aggregating grad dicts 2"):
    #             name = f.replace("_2.pt", "")
    #             merged_tensor = None
    #             for r in range(world_size):
    #                 t = torch.load(os.path.join(save_file_dir, f"rank{r}", f))
    #                 if merged_tensor is None:
    #                     merged_tensor = t
    #                 else:
    #                     merged_tensor += t
    #             merged_grad_2[name] = merged_tensor

    #         # 保存最终结果
    #         save_grad_epoch(merged_grad, merged_grad_2, save_dir=save_file_dir)

    #     dist.barrier()
   
    def train_one_task(self, task, i_task, epochs):
        # 在单独某个任务上训练
        print_rank_0(
            f"***** Task: {task} *****",
            self.args.global_rank)
        
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)

        save_file_dir = os.path.join(self.args.mask_save_file_dir, f"{task}_Fisher", "parameters_grad")
        grad_dir = self.process_mask(task, save_file_dir, device)
        # grad_dir = "/data1/TAP/model_con/baseline_14B/SPTCF/C-STANCE_Fisher/parameters_grad_2/epoch0"
        if self.args.local_rank == 0:
            self.args.mask_path = generation_mask_file(self.args, grad_dir)
        else:
            self.args.mask_path = None
        obj_list = [self.args.mask_path]
        dist.broadcast_object_list(obj_list, src=0)
        self.args.mask_path = obj_list[0]
        dist.barrier()
        
        if not hasattr(self, "grad_hook_handles"):
            self.grad_hook_handles = {}
        if self.args.mask_path is not None:
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            for name, param in model_ref.named_parameters():
                if param.requires_grad:
                    handle = param.register_hook(capture_grad_hook(name, self.args.mask_path))
                    self.grad_hook_handles[name] = handle
                    
        print_rank_0(
            f"***** Task: {task}  hook finish*****",
            self.args.global_rank)
        #### TRAIN ####
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                self.model.step()
                if (step + 2) % self.args.gradient_accumulation_steps == 0:
                    self.eval_on_dataset(epoch, epochs, step, task, device, total_steps)
            self.save_task_model(task, epoch + 1)
            
        for name, handle in list(self.grad_hook_handles.items()):
            try:
                handle.remove()
            except Exception:
                pass
        self.grad_hook_handles.clear()

            # Evaluate perplexity on the validation set.
            # print_rank_0(
            #     f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs} *****",
            #     self.args.global_rank)
            # perplexity = self.perplexity_evaluation(eval_dataloader, device)
            # print_rank_0(f"ppl: {perplexity}", self.args.global_rank)
            # self.model.tput_timer.update_epoch_count()
        
