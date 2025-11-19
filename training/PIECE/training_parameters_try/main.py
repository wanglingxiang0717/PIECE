#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import sys
sys.dont_write_bytecode = True

import argparse
import os
import math
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, create_prompt_dataset_dict
from utils.data.data_collator import DataCollator
from utils.data.data_collator_pretrain import DataCollator_pt
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
replace_bloom_attn_with_flash_attn()

# my_peft中修改了lora相关的逻辑
from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model

from train_base_model import CL_Base_Model

from params import Method2Class, AllDatasetName


# TODO, check support for OPT and llama


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    # parser.add_argument('--dataset_name',
    #                     type=list_of_strings,
    #                     default='all',
    #                     help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    # added by wangxiao
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')
    
    #added by wanglingxiang
    parser.add_argument('--shuffle',
                action='store_true',
                help='Sequential or Random')
    
    parser.add_argument('--is_pretrin',
                action='store_true',
                help='Sequential or Random')
    
    parser.add_argument('--eval_on_train_dataset',
                action='store_true',
                help='Evaluation on train dataset')
    
    parser.add_argument('--mask_path',
            default=None,
            help='grad mask file path')
    
    parser.add_argument('--param_import_savepath',
            default=None,
            help='grad mask file path')
    
    parser.add_argument("--ood_eval_dir",
                    type=str,
                    default=None,
                    help="The Out-of-Distribution evaluattion datasets.")
    parser.add_argument("--base_model",
            type=str,
            help="The top_ratio selection.")   
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        global_rank = 0
    else:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

    args.global_rank = global_rank

    set_random_seed(args.seed)
    if args.local_rank != -1:
        dist.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    model = create_hf_model(AutoModelForCausalLM,
                            args.base_model,
                            tokenizer,
                            ds_config=None,   # 不再传 deepspeed 配置
                            disable_dropout=args.disable_dropout,
                            )
    if args.base_model != args.model_name_or_path:
        state_dict_bin = torch.load(f"{args.model_name_or_path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict_bin)   
    model.to(device)

    # embedding_layer = model.get_input_embeddings()
    # for p in embedding_layer.parameters():
    #     p.requires_grad = False

    train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_output_path,
        args.seed,
        distributed=False,
    )

    if args.local_rank == -1 and args.shuffle:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
        test_sampler = SequentialSampler(test_dataset)
    elif args.local_rank == -1:
        train_sampler = SequentialSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
        test_sampler = SequentialSampler(test_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
        test_sampler = DistributedSampler(test_dataset)

    data_collator = DataCollator(
        tokenizer,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=False
    )
    inf_data_collator = DataCollator(
        tokenizer,
        model=model,
        padding="longest",
        max_prompt_len=args.max_prompt_len,
        max_ans_len=args.max_ans_len,
        pad_to_multiple_of=8,
        inference=True
    )

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset,
                                 collate_fn=inf_data_collator,
                                 sampler=test_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.gradient_checkpointing:
        try:
            model.module.gradient_checkpointing_enable() 
        except Exception:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    CL_Trainer = CL_Base_Model(model, tokenizer, train_dataloader, eval_dataloader, test_dataloader, args)
    CL_Trainer.train()


if __name__ == "__main__":
    main()
