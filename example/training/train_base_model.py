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
from transformers import GenerationConfig
from torch.utils.tensorboard import SummaryWriter
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
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

class Base_Model:
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

        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank)
            self.model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                if "cache_position" in batch:
                    batch.pop("cache_position")
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)

                loss = outputs.loss

                if self.args.global_rank == 0:
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
                    
                    ppl_eval, losses_eval = self.perplexity_evaluation(eval_dataloader, device)
                    print_rank_0(f"[eval loss, ppl] step:{(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s}," 
                                 f"\tloss: {losses_eval}, \tppl: {ppl_eval}", 
                                 self.args.global_rank)

                    writer.add_scalar('eval/ppl', ppl_eval, 
                                        global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                    writer.add_scalar('eval/loss', losses_eval, 
                                        global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                    if self.eval_dataloader_dict is not None:
                        for item in self.eval_dataloader_dict:
                            ppl, losses = self.perplexity_evaluation(self.eval_dataloader_dict[item], device)
                            print_rank_0(f"[{item} loss, ppl] step:{(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s},"
                                        f"\tloss: {losses}, \tppl: {ppl}", self.args.global_rank) 
                            writer.add_scalar(f'{item}/ppl', ppl, 
                                              global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                            writer.add_scalar(f'{item}/loss', losses, 
                                              global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
            
            self.save_model(epoch + 1)

    def save_model(self, round):
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
        
