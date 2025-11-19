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

def save_gradient(name, step, epoch, grad, save_dir="/data2/TAP/data/save_grad/humaneval/"):
    save_dir_path = f"{save_dir}/epoch{epoch}_step{step}"
    os.makedirs(save_dir_path, exist_ok=True)
    filename = f"{save_dir_path}/epoch{epoch}_step{step}_{name}.pt"
    torch.save(grad, filename)
    
grad_dict = {}

def capture_grad_hook(name):
    def hook(grad):
        grad_dict[name] = grad.detach().cpu().clone()
    return hook

    # def hook_fn(grad):
    #     return grad * mask  #通过梯度掩码的方式对某些参数实现保护(冻结)

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
        
        
    def perplexity_evaluation(self, eval_dataloader, device):
        # 验证集上测困惑度
        self.model.eval()
        losses = 0
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
        except:
            pass
        self.model.train()
        return perplexity, losses


    def train(self):
        # 在单独某个任务上训练
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
        eval_dataloader2 = self.eval_dataloader2
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        
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
                # Update the description to include current step and loss, if needed
                if self.args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                self.model.backward(loss)

                # for name, grad in grad_dict.items():
                #     # print(f'{name} grad is {grad is not None}')
                #     save_gradient(name, step, epoch, grad)
                # grad_dict.clear()
                
                # self.model.zero_grad() #模型参数不更新，梯度置为0
                self.model.step()                 

                # Evaluate perplexity on the validation set.
                # if (step + 2) % self.args.gradient_accumulation_steps == 0:
                #     g_s = (step + 2) / self.args.gradient_accumulation_steps - 1
                #     print_rank_0(
                #         f"***** Evaluating perplexity, Epoch {epoch+1}/{epochs}, step {g_s} *****",
                #         self.args.global_rank)
                #     # perplexity, losses = self.perplexity_evaluation(train_dataloader, device)
                #     # print("[train loss, ppl]", "step:", (epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s, "\tloss:", losses, "\tppl:", perplexity)
                #     ppl_eval, losses_eval = self.perplexity_evaluation(eval_dataloader, device)
                #     print("[eval loss, ppl]", "step:", (epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s, "\tloss:", losses_eval, "\tppl:", ppl_eval)
                #     ppl_eval2, losses_eval2 = self.perplexity_evaluation(eval_dataloader2, device)
                #     print("[eval2 loss, ppl]", "step:", (epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s, "\tloss:", losses_eval2, "\tppl:", ppl_eval2)
                #     # writer.add_scalar('train/ppl', perplexity, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                #     # writer.add_scalar('train/all_loss', losses, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                #     writer.add_scalar('eval/ppl', ppl_eval, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                #     writer.add_scalar('eval/loss', losses_eval, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                #     writer.add_scalar('eval2/ppl', ppl_eval2, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
                #     writer.add_scalar('eval2/loss', losses_eval2, global_step=(epoch * len(train_dataloader)) / self.args.gradient_accumulation_steps + g_s)
            
            self.save_model(epoch + 1)
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
        
