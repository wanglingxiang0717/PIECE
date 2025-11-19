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

generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)

grad_dict = {}
grad_dict_2 = {}

def capture_grad_hook(name):
    def hook(grad):
        if grad is None:
            return
        grad_sq = grad.detach().cpu().double()
        grad_sq = torch.nan_to_num(grad_sq, nan=0.0)
        grad_sq_2 = grad_sq ** 2

        # if torch.isnan(grad_sq).any():
        #     print(f"[Warning] NaN detected in gradient of {name}, replacing with 0.")
        #     grad_sq = torch.nan_to_num(grad_sq, nan=0.0)
        #     grad_sq_2 = grad_sq ** 2
        if name in grad_dict:
            grad_dict[name] += grad_sq
        else:
            grad_dict[name] = grad_sq.clone()

        if name in grad_dict_2:
            grad_dict_2[name] += grad_sq_2
        else:
            grad_dict_2[name] = grad_sq_2.clone()
    return hook


def save_grad_epoch(epoch, save_dir="/data2/TAP/data/save_grad/humaneval/"):
    save_dir_path = f"{save_dir}/epoch{epoch}"
    save_dir_path_2 = f"{save_dir.replace('parameters_grad', 'parameters_grad_2')}/epoch{epoch}"
    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(save_dir_path_2, exist_ok=True)

    if dist.is_initialized():
        for name in list(grad_dict.keys()):
            tensor = grad_dict[name].to("cuda")  # all_reduce 必须在 GPU 上
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            grad_dict[name] = tensor.cpu()

        for name in list(grad_dict_2.keys()):
            tensor = grad_dict_2[name].to("cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            grad_dict_2[name] = tensor.cpu()
            
    if not dist.is_initialized() or dist.get_rank() == 0:
        for name, grad_sq in grad_dict.items():
            filename = f"{save_dir_path}/{name}.pt"
            torch.save(grad_sq, filename)

        for name, grad_sq_2 in grad_dict_2.items():
            filename = f"{save_dir_path_2}/{name}.pt"
            torch.save(grad_sq_2, filename)

    grad_dict.clear()
    grad_dict_2.clear()


class CL_Base_Model:
    def __init__(self,
                 model,
                 tokenizer,
                 optimizer,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 eval_dataloader2,
                 args):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.eval_dataloader2 = eval_dataloader2
        self.args = args

    def train(self):
        # 在单独某个任务上训练
        writer = SummaryWriter(log_dir=self.args.tensorboard_path)
        epochs = int(self.args.num_train_epochs[0])
        if self.args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            device = torch.device("cuda", self.args.local_rank)
        
        #### TRAIN ####
        train_dataloader = self.train_dataloader
        eval_dataloader = self.eval_dataloader
        eval_dataloader2 = self.eval_dataloader2
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
        # 注册 hook 捕获梯度
        for name, param in self.model.module.named_parameters():
            if param.requires_grad:
                param.register_hook(capture_grad_hook(name))
                
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

                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)
                self.model.zero_grad()  # 模型参数不更新，梯度置为0
            save_grad_epoch(epoch, save_dir=self.args.param_import_savepath)            
            
            # self.save_model(epoch + 1)
            
            # save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(epoch + 1))
    
    
    # def train_continual(self):
    #     for i_task, task in enumerate(self.train_task_list):
    #         self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))
    #         self.save_model(i_task)

    
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
        
