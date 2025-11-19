import numpy as np
import torch
import quadprog
import random
from tqdm.auto import tqdm
import copy
import json
import torch.nn.functional as F
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, save_hf_format, save_hf_format_task, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset
import gc

class LwF(CL_Base_Model):
    # def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args
    #              ):
    #     super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
    def __init__(self,model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args)
        # self.device = torch.device("cuda" if args.local_rank == -1 else f"cuda:{args.local_rank}")
        # if args.local_rank != -1:
        #     torch.cuda.set_device(self.device)
            
    def train_step(self,
                    batch):

        lm_labels = batch["labels"]
        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'],use_cache=False)
        return outputs
    
    
    def KD_loss(self, new_logits, prev_logits, T):
        prev_logits = torch.from_numpy(prev_logits).to(torch.bfloat16)
        prev_logits = prev_logits.to('cuda')
        kd_loss = F.kl_div(
            F.log_softmax(new_logits / T, dim=-1),   # student log-prob
            F.softmax(prev_logits / T, dim=-1),      # teacher prob
            reduction='batchmean'
        ) * (T * T)
        return kd_loss

    # def KD_loss(self, new_logits, prev_topk, T):
    #     prev_values, prev_indices = prev_topk
    #     prev_values = prev_values.to(new_logits.device, dtype=torch.float16)
    #     prev_indices = prev_indices.to(new_logits.device)

    #     # 还原成稀疏分布
    #     prev_probs = torch.zeros_like(new_logits, dtype=torch.float16)
    #     prev_probs.scatter_(dim=-1, index=prev_indices, src=F.softmax(prev_values / T, dim=-1))

    #     kd_loss = F.kl_div(
    #         F.log_softmax(new_logits / T, dim=-1),
    #         prev_probs,
    #         reduction='batchmean'
    #     ) * (T * T)
    #     return kd_loss

    def new_input_old_model_logits(self, i_task):
        task_name = list(self.train_task_list.keys())[i_task+1]
        train_dataloader = self.train_task_list[task_name]
        self.new_task_logits = {}
        for step, batch in enumerate(train_dataloader):
            del batch['sources']
            batch = {k:batch[k].to('cuda') for k in batch}
            outputs = self.train_step(batch)
            # logits = outputs.logits.to(torch.float32).detach().cpu().numpy()
            logits = outputs.logits.to(torch.float16).detach().cpu().numpy()
            self.new_task_logits[str(step)] = logits
            del logits
            del outputs

    # @torch.inference_mode()
    # def new_input_old_model_logits(self, i_task):
    #     task_name = list(self.train_task_list.keys())[i_task + 1]
    #     train_dataloader = self.train_task_list[task_name]
    #     self.new_task_logits = {}

    #     for step, batch in enumerate(train_dataloader):
    #         batch.pop("sources", None)
    #         batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
    #         outputs = self.train_step(batch)
    #         logits = outputs.logits.to(torch.float16).cpu()
    #         topv, topi = torch.topk(logits, k=64, dim=-1)
    #         self.new_task_logits[str(step)] = (topv, topi)

    #         del logits, outputs, batch
    #         torch.cuda.empty_cache()
    #         gc.collect()

    def train_one_task(self,
                       task,
                       i_task,
                       epochs=40):

        # print('task = ', task)
        print_rank_0(
            f"***** Task: {task} *****",
            self.args.global_rank)
        
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
      
        train_dataloader = self.train_task_list[task]
        
        if i_task == 0:
            print_rank_0(
                f"Preparing ……",
                self.args.global_rank)
            self.new_task_logits = {}
            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = {k: batch[k].to('cuda') for k in batch if k != 'sources'}
                outputs = self.train_step(batch)
                # logits = outputs.logits.to(torch.float32).detach().cpu().numpy()
                logits = outputs.logits.to(torch.float16).detach().cpu().numpy()
                self.new_task_logits[str(step)] = logits
                del logits
                del outputs
                
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()
            total_steps = epochs * len(train_dataloader)
            progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                # batch = {k:batch[k].to('cuda') for k in batch}
                batch = to_device(batch, device)
                outputs = self.train_step(batch)
                loss = outputs.loss
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                    
                # if i_task!=0:
                loss =0.5 * loss + 0.5 * self.KD_loss(outputs.logits, self.new_task_logits[str(step)], 2)
                
                self.model.backward(loss)
                self.model.step()
                if (step + 2) % self.args.gradient_accumulation_steps == 0:
                    self.eval_on_dataset(epoch, epochs, step, task, device, total_steps)
            # self.save_task_model(task, epoch + 1)
            self.eval_model_generation(task, epoch, epochs)
                
        if i_task+1 < len(self.train_task_list):
            self.new_input_old_model_logits(i_task)

            
