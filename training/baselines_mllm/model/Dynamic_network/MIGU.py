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
from utils.process_mask import generation_mask
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)

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
# MASK_CACHE = {}

# def capture_grad_hook(name, file_path=None):
#     def hook_fn(grad):
#         if name not in MASK_CACHE:
#             if file_path:
#                 json_file = f"{file_path}/{name}.json"
#                 if os.path.exists(json_file):
#                     with open(json_file, 'r') as f:
#                         data = json.load(f)
#                         MASK_CACHE[name] = data["positions"]
#                 else:
#                     MASK_CACHE[name] = None  
#             else:
#                 MASK_CACHE[name] = None

#         positions = MASK_CACHE[name]

#         if positions is None:
#             return grad
#         else:
#             mask = torch.zeros_like(grad, device="cpu")
#             try:
#                 for pos in positions:
#                     mask[tuple(pos)] = 1.0
#             except IndexError:
#                 print(f"[警告] 索引超界: {name} grad.shape={grad.shape}")

#         mask = mask.to(grad.device, non_blocking=True)

#         out = grad * mask
#         del mask  
#         return out

#     return hook_fn

class MIGU(CL_Base_Model):
    def __init__(self,model, tokenizer, processor, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args, lambda_ewc=400):
        super().__init__(model, tokenizer, processor, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args)


    def process_mask_file(self, task, save_file_dir):
        self.model.eval()
        local_mean_dict, shape_len = generation_mask(self.model, self.tokenizer, task, self.args.data_path, self.args)
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        global_mean_dict = {}
        for layer_id, local_tensor in local_mean_dict.items():
            tensor_copy = local_tensor.clone()
            dist.all_reduce(tensor_copy, op=dist.ReduceOp.SUM)
            tensor_copy /= world_size
            global_mean_dict[layer_id] = tensor_copy

        # rank 0 保存结果
        if rank == 0:
            # stop = input(global_mean_dict)
            top_ratio = self.args.mask_top_ratio
            mask_dict = {}
            if not os.path.exists(save_file_dir):
                os.makedirs(save_file_dir)

            for layer_id, mean in global_mean_dict.items():
                k = max(1, int(len(mean) * top_ratio))
                threshold = torch.topk(mean, k).values.min()
                mask = (mean >= threshold)  # bool tensor
                mask_dict[layer_id] = mask.nonzero(as_tuple=False).squeeze()
            
            for name, param in self.model.module.named_parameters():
                name_list = name.split(".")
                if "layers" in name_list and param.shape[-1] == shape_len:
                    save_file = f"{save_file_dir}/{name}.json"
                    position_list = []
                    layer_id = int(name_list[2])
                    mask_list = mask_dict[layer_id]
                    for i in mask_list:
                        if param.dim() == 1:
                            pos = [i.item()]
                            position_list.append(pos)
                        else:
                            assert param.dim() == 2
                            for j in range(param.shape[0]):
                                pos = [j, i.item()]
                                position_list.append(pos)
                    with open(save_file, "w") as f:
                        json.dump({"positions": position_list}, f)
        dist.barrier()
        self.model.train()
        
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
        
        # if self.args.CL_method == "MIGU":
        save_file_dir = os.path.join(self.args.mask_save_file_dir, f"{task}_mask")
        self.process_mask_file(task, save_file_dir)
        if not hasattr(self, "grad_hook_handles"):
            self.grad_hook_handles = {}

        for name, handle in list(self.grad_hook_handles.items()):
            try:
                handle.remove()
            except Exception:
                pass
        self.grad_hook_handles.clear()
        
        if save_file_dir is not None:
            model_ref = self.model.module if hasattr(self.model, "module") else self.model
            for name, param in model_ref.named_parameters():
                if param.requires_grad:
                    handle = param.register_hook(capture_grad_hook(name, save_file_dir))
                    self.grad_hook_handles[name] = handle

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
                if 'sources' in batch:
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
            # self.save_task_model(task, epoch + 1)
            self.eval_model_generation(task, epoch, epochs)
            


            # Evaluate perplexity on the validation set.
            # print_rank_0(
            #     f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs} *****",
            #     self.args.global_rank)
            # perplexity = self.perplexity_evaluation(eval_dataloader, device)
            # print_rank_0(f"ppl: {perplexity}", self.args.global_rank)
            # self.model.tput_timer.update_epoch_count()
        
