import torch
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
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
from torch.utils.tensorboard import SummaryWriter
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from utils.test_model import generation_file
from utils.get_sample import process_get_result_file
from utils.test_model_safe import generation_safe_file
from human_eval.data import write_jsonl, read_problems
generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)
    
grad_dict = {}

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

class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 eval_dataloader_dict,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.eval_dataloader_dict = eval_dataloader_dict
        self.args = args
        
        
    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                # implementation, batch = {k: v.to(device) for k, v in batch.items()}
                del batch['sources']
                batch = to_device(batch, device)
                with torch.no_grad():
                    outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                losses += loss.float()
            losses = losses / (step + 1)
            try:
                perplexity = torch.exp(losses)
            except OverflowError:
                perplexity = float("inf")
            try:
                perplexity = get_all_reduce_mean(perplexity).item()
                loss = get_all_reduce_mean(loss).item()
            except:
                pass
        self.model.train()
        return perplexity, losses


    def train(self):
        writer = SummaryWriter(log_dir=self.args.tensorboard_path)
        epochs=int(self.args.num_train_epochs[0])
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_dataloader
        eval_dataloader = self.eval_dataloader
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        ##hook注册
        if self.args.mask_path is not None:
            # stop = input(self.args.mask_path)
            # if self.args.target_name not in ["MeetingBank", "ScienceQA", "Py150", "20Minuten"]:
            #     for name, param in self.model.module.named_parameters():
            #         if param.requires_grad:
            #             param.register_hook(capture_grad_hook(name, self.args.mask_path))
            # else:
            for name, param in self.model.module.named_parameters():
                if param.requires_grad:
                    param.register_hook(capture_grad_hook(name, self.args.mask_path))
               

        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                # stop = input(batch)
                if "cache_position" in batch:
                    batch.pop("cache_position")
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                # outputs = self.model(**batch)
                # stop = input(outputs)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                self.model.step()                 

                # Evaluate perplexity on the validation set.
                if (step + 2) % self.args.gradient_accumulation_steps == 0:
                    g_s = (step + 2) / self.args.gradient_accumulation_steps - 1
                    print_rank_0(
                        f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs}, step {g_s} *****",
                        self.args.global_rank)
                    
                    # if self.args.eval_on_train_dataset:
                    #     perplexity, losses = self.perplexity_evaluation(train_dataloader, device)
                    #     print_rank_0(f"[train loss, ppl], step: {(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s}, \tloss: losses \tppl: perplexity")
                    #     writer.add_scalar('train/ppl', perplexity, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                    #     writer.add_scalar('train/all_loss', losses, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                    ppl_eval, losses_eval = self.perplexity_evaluation(eval_dataloader, device)
                    print_rank_0(f"[eval loss, ppl] step:{(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s}, \tloss: {losses_eval}, \tppl: {ppl_eval}", 
                                 self.args.global_rank)
                    # print("[eval loss, ppl]", "step:", (epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s, "\tloss:", losses_eval, "\tppl:", ppl_eval)
                    
                    writer.add_scalar('eval/ppl', ppl_eval, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                    writer.add_scalar('eval/loss', losses_eval, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                    if self.eval_dataloader_dict is not None:
                        for item in self.eval_dataloader_dict:
                            ppl, losses = self.perplexity_evaluation(self.eval_dataloader_dict[item], device)
                            # print_rank_0(f"[{item} loss, ppl]", "step:", (epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s, "\tloss:", losses, "\tppl:", ppl)
                            print_rank_0(f"[{item} loss, ppl] step:{(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s}, \tloss: {losses}, \tppl: {ppl}", 
                                                            self.args.global_rank) 
                            writer.add_scalar(f'{item}/ppl', ppl, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                            writer.add_scalar(f'{item}/loss', losses, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
            
            # self.eval_model_generation(task, epoch, epochs)
            self.save_model(epoch + 1)
            # save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(epoch + 1))

    def eval_model_generation(self, task, epoch, epochs):
        local_results = generation_file(task, self.args.dataset_name, self.model, self.tokenizer, 
                        self.args.test_file_dir, os.path.join(self.args.output_dir, task) + "/" + str(epoch + 1))
        dist.barrier()
        if dist.get_rank() == 0:
            all_results = [None for _ in range(dist.get_world_size())]
        else:
            all_results = None

        dist.gather_object(local_results, all_results, dst=0)

        if dist.get_rank() == 0:
            merged = merge_generation_results(all_results)
            save_dir = os.path.join(self.args.output_dir, task) + "/" + str(epoch + 1)
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)
            for task in merged:
                file_save_path = f"{save_dir}/{task}_multibatch_result.json"
                write_json_or_jsonl(file_save_path, merged[task])
            
        # if epoch + 1 == epochs and self.args.global_rank == 0:
        if epoch + 1 == epochs:
            local_samples = process_get_result_file(self.model, self.tokenizer)
            dist.barrier()
            all_samples = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
            dist.gather_object(local_samples, all_samples, dst=0)
            merged = []
            if dist.get_rank() == 0:
                for part in all_samples:
                    merged.extend(part)
                humaneval_save_dir = os.path.join(self.args.output_dir, task) + "/" + str(epoch + 1)
                write_jsonl(f"{humaneval_save_dir}/humaneval_dosample_test.jsonl", merged)
                
            local_result_safe = generation_safe_file(self.model, self.tokenizer, self.args.test_file_dir)
            dist.barrier()
            all_samples_safe = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
            dist.gather_object(local_result_safe, all_samples_safe, dst=0)
            merged_safe = []
            if dist.get_rank() == 0:
                for part in all_samples_safe:
                    merged_safe.extend(part)
                safe_save_dir = os.path.join(self.args.output_dir, task) + "/" + str(epoch + 1)
                # write_jsonl(f"{humaneval_save_dir}/humaneval_dosample_test.jsonl", merged)
                write_json_or_jsonl(f"{safe_save_dir}/MaliciousInstruct_multibatch_result.json", merged_safe) 
        self.model.train()  
    
    # def train_continual(self):
    #     for i_task, task in enumerate(self.train_task_list):
    #         self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
    #         self.save_model(i_task)

    
    def save_model(self, round):
        # stop = input(self.model.config)
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))

        if self.args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Sucessful saving model after epoch {}'.format(round), self.args.global_rank)
        
