from copy import deepcopy

import torch
from torch.autograd import Variable
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, save_hf_format, save_hf_format_task, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.data.data_utils import create_prompt_dataset

class EWC(CL_Base_Model):
    def __init__(self,model, tokenizer, processor, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args, lambda_ewc=400):
        super().__init__(model, tokenizer, processor, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args)
        self.device="cuda"
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._previous_params = {}

        for n, p in deepcopy(self.params).items():
            self._previous_params[n] = p.data.cpu() # Previous task parameters
        self.grads = {} # 存储节点名称与节点的grad
        
        self.fisher = {}
        self.init_fisher()
        del self.params
        
    def init_fisher(self):
        for n, p in deepcopy(self.params).items():
            if p.requires_grad==True:
                p.data.zero_()  #所有参数置零
                self.fisher[n] = p.data  #初始化零矩阵
    # def init_fisher(self):
    #     for n, p in deepcopy(self.params).items():
    #         if p.requires_grad:
    #             self.fisher[n] = torch.zeros_like(p.data, device="cpu")
            
    #计算每个参数的Fisher信息矩阵的值：每个样本输入模型，每个参数计算梯度的平方和，除以总的样本数量
    def _update_fisher(self):
        for n, p in self.model.named_parameters():
            if n in self.grads.keys():
                self.fisher[n].data += self.grads[n].cuda().data ** 2 / self.train_length
    # def _update_fisher(self):
    #     for n, p in self.model.named_parameters():
    #         if n in self.grads.keys():
    #             self.fisher[n] += (self.grads[n] ** 2) / self.train_length
    #正则化，除以训练集长度
    def _regular_fisher(self):
        for n, p in self.model.named_parameters():
            if n in self.grads.keys():
                self.fisher[n].data /= self.train_length

    
    def _update_previous_params(self):
        for n, p in self.model.named_parameters():
            self._previous_params[n] = p.data.cpu() # Previous task parameters

    #计算惩罚loss
    def penalty(self):
        restrict_loss = 0
        precision_matrices = self.fisher
        for n, p in self.model.named_parameters():
            if p.requires_grad==True:
                restrict_loss_params = precision_matrices[n] * (p - self._previous_params[n].cuda()) ** 2
                restrict_loss += restrict_loss_params.sum()
        return restrict_loss
    # def penalty(self):
    #     restrict_loss = 0
    #     for n, p in self.model.named_parameters():
    #         if p.requires_grad:
    #             fisher = self.fisher[n].to(p.device)  # 临时搬上 GPU
    #             prev_param = self._previous_params[n].to(p.device)
    #             restrict_loss_params = fisher * (p - prev_param) ** 2
    #             restrict_loss += restrict_loss_params.sum()
    #     return restrict_loss
    
    def train_step(self,
                    batch):

        # batch = {k: batch[k].to(self.device) for k in batch}
        lm_labels = batch["labels"]
        # lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        # inputs_embeds = model.encoder.embed_tokens(batch["source_ids"])
        # inputs_embeds = self.model.model.embed_tokens(batch["input_ids"])  #向量，【batch * embedding_size】

        outputs = self.model(input_ids=batch['input_ids'], labels=lm_labels, attention_mask=batch['attention_mask'],use_cache=False)
        
        loss = outputs[0]
        if self.task_num!=0:
            restrict_loss = self.penalty()
            loss += 0.5*self.lambda_ewc*restrict_loss

        return loss
    
    def save_grad(self,name):
        def hook(grad):
            grad = torch.nan_to_num(grad, nan=0)
            # grad = torch.clamp(grad, -self.args.ds_config['gradient_clipping'], self.args.ds_config['gradient_clipping'])
            self.grads[name] = grad.cpu()  #这里得考虑将其改成GPU上，但是考虑可能会爆显存
            del grad
        return hook
    
    def retain_grad(self):
        # print("begin")
        for n,p in self.model.named_parameters():
            if n in self.fisher.keys():
                p.register_hook(self.save_grad(n))
        # print("finish")
    
    
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
        self.train_length = len(train_dataloader)
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
                # batch = {k:batch[k].to('cuda') for k in batch}
                batch = to_device(batch, device)
                loss = self.train_step(batch)
                
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)
                self.model.backward(loss)
                self.model.step()
                self._update_fisher()
                if (step + 2) % self.args.gradient_accumulation_steps == 0:
                    self.eval_on_dataset(epoch, epochs, step, task, device, total_steps)
            # self.save_task_model(task, epoch + 1)
            self.eval_model_generation(task, epoch, epochs)
    
    # Train model continually
    def train_continual(self):
        #在训练之前确定梯度
        self.retain_grad()

        for i_task, task in enumerate(self.train_task_list):
            self.task_num=i_task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
            # self._regular_fisher()
            
            self._update_previous_params()
            # self.save_model(i_task)
            
            


