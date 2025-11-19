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
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args, lambda_ewc=400):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, eval_dataloader_dict, args)
        self.device = "cuda"
        self.lambda_ewc = lambda_ewc

        # 仅保存需要梯度的参数
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        # 历史参数和 Fisher 信息都放 CPU
        self._previous_params = {n: p.data.cpu().clone() for n, p in self.params.items()}
        self.fisher = {n: torch.zeros_like(p.data, device="cpu") for n, p in self.params.items()}
        self.grads = {}

        del self.params  # 释放引用

    def save_grad(self, name):
        def hook(grad):
            grad = torch.nan_to_num(grad.detach(), nan=0).cpu()  # 直接保存到 CPU
            self.grads[name] = grad
            del grad
        return hook

    def retain_grad(self):
        # 注册梯度 hook，只保存必要梯度
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                p.register_hook(self.save_grad(n))

    @torch.no_grad()
    def _update_fisher(self):
        """梯度平方和累积到 CPU 上的 Fisher"""
        for n, p in self.model.named_parameters():
            if n in self.grads:
                self.fisher[n] += (self.grads[n] ** 2) / self.train_length
        self.grads.clear()  # 清空防止占显存

    @torch.no_grad()
    def _update_previous_params(self):
        """更新上一任务参数（放 CPU）"""
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._previous_params[n] = p.data.cpu().clone()

    def penalty(self):
        """惩罚项在 CPU 上逐层计算，只把当前层搬上 GPU"""
        restrict_loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                fisher_cpu = self.fisher[n]
                prev_param_cpu = self._previous_params[n]

                # 临时搬上 GPU，参与运算
                fisher = fisher_cpu.to(p.device, non_blocking=True)
                prev_param = prev_param_cpu.to(p.device, non_blocking=True)

                restrict_loss += (fisher * (p - prev_param) ** 2).sum()

                # 手动释放
                del fisher, prev_param

        torch.cuda.empty_cache()  # 清理临时缓存
        return restrict_loss

    def train_step(self, batch):
        batch = to_device(batch, self.device)
        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            attention_mask=batch['attention_mask'],
            use_cache=False,
        )

        loss = outputs[0]
        if self.task_num != 0:
            restrict_loss = self.penalty()
            loss += 0.5 * self.lambda_ewc * restrict_loss
        return loss

    def train_one_task(self, task, i_task, epochs=40):
        print_rank_0(f"***** Task: {task} *****", self.args.global_rank)
        device = torch.device("cuda", self.args.local_rank if self.args.local_rank != -1 else 0)
        train_dataloader = self.train_task_list[task]

        self.train_length = len(train_dataloader)
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                loss = self.train_step(batch)

                progress_bar.update(1)
                if self.args.global_rank == 0:
                    progress_bar.set_description(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

                self.model.backward(loss)
                self.model.step()

                self._update_fisher()
                torch.cuda.empty_cache()  # 防止累计显存占用

            self.eval_model_generation(task, epoch, epochs)

            
            


