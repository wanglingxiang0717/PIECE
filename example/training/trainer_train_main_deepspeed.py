import sys
sys.dont_write_bytecode = True
import os
import json
import argparse
import torch
from torch.utils.data import Dataset
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from piece import process_mask, mask_grads

class SFTDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        assert "prompt" in self.data[0] and "answer" in self.data[0]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        answer = item["answer"]
        text = prompt + answer
        input_ids = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=True 
        )["input_ids"]
        prompt_input_ids = self.tokenizer(
            prompt, 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=True 
        )["input_ids"]
        
        labels = list(input_ids)
        mask_len = len(prompt_input_ids)
        for i in range(min(mask_len, len(labels))):
            labels[i] = -100
        
        return {
            "input_ids": input_ids, 
            "attention_mask": [1] * len(input_ids), 
            "labels": labels
        }

def train_sft(args):
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    train_path = os.path.join(args.data_dir, "train.json")
    assert os.path.exists(train_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.to(device)
    
    train_dataset = SFTDataset(train_path, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    save_dir = args.output_dir
    top_ratio = 0.001
    mode = "S"
    process_mask(model = model, 
                 dataset = train_dataset, 
                 data_collator = data_collator, 
                 save_dir = save_dir, 
                 mode = mode, 
                 top_ratio = top_ratio,
                 args = args)
    if args.local_rank != -1:
        torch.distributed.barrier()
    
    mask_grads(model=model, 
               mode = mode, 
               save_dir = save_dir,
               top_ratio = top_ratio)
     
    # DeepSpeed 配置
    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    ds_config = {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu"},
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": {"enabled": True},
        "fp16": {"enabled": False}
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=4,
        report_to="none",
        deepspeed=ds_config,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # 训练
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"训练完成，模型保存在：{args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--param_import_savepath", type=str, required=True)
    parser.add_argument("--top_ratio", type=float, default=0.001)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed by deepspeed")

    args = parser.parse_args()
    train_sft(args)
