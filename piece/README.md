# PIECE

<!-- PROJECT SHIELDS -->

<!-- [![Contributors][contributors-shield]][contributors-url] -->

<!-- [![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->

<!-- <br />

<p align="center">
  <a href="https://github.com/shaojintian/Best_README_template/">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">"PIECE</h3>
  <p align="center">
    PIECE
    <br />
    <a href="None"><strong>PIECE_doc</strong></a>
    <br />
    <br />
    <a href="None">Demo</a>
   ,
    <a href="None">Bug</a>
   ,
    <a href="None">Issues</a>
  </p>

</p> -->

# Table of Contents

* [Quick Start](#quick-start)
* [Simple Examples](#simple-examples)
* [Core Methods](#core-methods)
* [Partial Code Sources](#partial-code-sources)
* [Authors](#authors)

<!-- - [Acknowledgements](#acknowledgements) -->

---

## Quick Start

### Environment Setup

```bash
git clone https://github.com/wanglingxiang0717/PIECE.git
cd PIECE/piece
pip install -e . --no-build-isolation
```

### Usage

```python
from piece import process_mask, mask_grads

# model, dataset, and data_collator initialized
save_dir = args.output_dir
top_ratio = 0.001
mode = "S"  # ["S", "F"]

process_mask(
    model=model, 
    dataset=train_dataset, 
    data_collator=data_collator, 
    save_dir=save_dir, 
    mode=mode, 
    top_ratio=top_ratio,
    args=args
)

# optimizer creation before training
mask_grads(
    model=model, 
    mode=mode, 
    save_dir=save_dir,
    top_ratio=top_ratio
)

# training starts here
```

---

## Simple Examples

### ***simple_torch***

```bash
cd example
chmod +x scripts/torch_train.sh
sh scripts/torch_train.sh
```

**`example/training/torch_train_main_deepspeed.py`**

```python
from piece import process_mask, mask_grads  # 51

save_dir = args.output_dir  # 365
top_ratio = 0.001          # 366
mode = "S"                  # 367
process_mask(               # 368
    model=model,
    dataset=train_dataset,
    data_collator=data_collator,
    save_dir=save_dir,
    mode=mode,
    top_ratio=top_ratio,
    args=args
)
if args.local_rank != -1:   # 375
    torch.distributed.barrier()  # 376

mask_grads(                 # 434
    model=model,
    mode=mode,
    save_dir=save_dir,
    top_ratio=top_ratio
)
```

### ***simple_Trainer (Transformers)***

```bash
cd example
chmod +x scripts/trainer_train.sh
sh scripts/trainer_train.sh
```

**`example/training/trainer_train_main_deepspeed.py`**

```python
from piece import process_mask, mask_grads  # 12

save_dir = args.output_dir  # 73
top_ratio = 0.001          # 74
mode = "S"                  # 75
process_mask(               # 76
    model=model,
    dataset=train_dataset,
    data_collator=data_collator,
    save_dir=save_dir,
    mode=mode,
    top_ratio=top_ratio,
    args=args
)
if args.local_rank != -1:   # 83
    torch.distributed.barrier()  # 84

mask_grads(                 # 86
    model=model,
    mode=mode,
    save_dir=save_dir,
    top_ratio=top_ratio
)
```

### ***simple_llama_factory***

`llama_factory` is not specially adapted yet. Training logic is mainly in:

```
LLaMA-Factory/src/llamafactory/train/{method}/workflow.py
```

Example for `sft` (replace source file with `example/training/workflow.py`):

```python
from piece import process_mask, mask_grads # 30

save_dir = training_args.output_dir # 70
top_ratio = 0.001
mode = "S"
train_dataset = dataset_module["train_dataset"]

process_mask(
    model=model, 
    dataset=train_dataset, 
    data_collator=data_collator, 
    save_dir=save_dir, 
    mode=mode, 
    top_ratio=top_ratio,
    args=training_args,
    cpu_offload=True
)
if training_args.local_rank != -1:
    torch.distributed.barrier()

mask_grads(
    model=model,   # 86
    mode=mode, 
    save_dir=save_dir,
    top_ratio=top_ratio
)
```

---

## Core Methods

### `process_mask`

```python
def process_mask(
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        data_collator,
        save_dir: str,
        mode: str,
        top_ratio: float,
        args: argparse.Namespace,
        singleGPU: bool = True,
        cpu_offload: bool = False,
        loss_function: Callable[
            [torch.nn.Module, Any, torch.device, argparse.Namespace], 
            torch.Tensor
        ] = standard_loss_function,
    ):
```

**Description:**
Processes and saves parameter masks for a model, used to selectively retain important parameters or apply gradient masking.

**Parameters:**

| Parameter       | Type                                                   | Description                                                                                 |
| --------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `model`         | `torch.nn.Module`                                      | The model whose parameters will be masked.                                                  |
| `dataset`       | `torch.utils.data.Dataset`                             | Dataset used for gradient computation.                                                      |
| `data_collator` | any                                                    | Data batching function (e.g., Hugging Face's `DataCollatorForSeq2Seq`).                     |
| `save_dir`      | `str`                                                  | Directory to save mask files (intermediate and final results).                              |
| `mode`          | `str`                                                  | Masking mode, either `'F'` (Fisher) or `'S'` (Second-order normalization).                  |
| `top_ratio`     | `float`                                                | Fraction of top parameters to keep (e.g., `0.001` = top 0.1%).                              |
| `args`          | `argparse.Namespace`                                   | Configuration arguments (e.g., `local_rank`, distributed training settings).                |
| `singleGPU`     | `bool`, optional, default `True`                       | Whether to compute gradients on a single GPU (recommended for safety).                      |
| `cpu_offload`   | `bool`, optional, default `False`                      | Whether to offload gradient computation to CPU to save GPU memory (slower).                 |
| `loss_function` | `Callable`, optional, default `standard_loss_function` | Custom loss function with signature: <br>`loss = loss_function(model, batch, device, args)` |

**Example:**

```python
process_mask(
    model=model,
    dataset=train_dataset,
    data_collator=data_collator,
    save_dir="./mask_output",
    mode="S",
    top_ratio=0.001,
    args=args
)
```

---

### `mask_grads`

```python
def mask_grads(
        model: torch.nn.Module,
        mode: str,
        save_dir: str,
        top_ratio: float,
        mask_file_path: str = None
    ):
```

**Description:**
Registers gradient hooks on model parameters to selectively mask gradients for later processing or saving.

**Parameters:**

| Parameter        | Type              | Description                                                                                                                                             |
| ---------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | `torch.nn.Module` | Model whose parameters will have gradient hooks registered.                                                                                             |
| `mode`           | `str`             | Masking mode, `'F'` or `'S'`.                                                                                                                           |
| `save_dir`       | `str`             | Directory to store mask information (intermediate and final).                                                                                           |
| `top_ratio`      | `float`           | Fraction of top parameters to keep.                                                                                                                     |
| `mask_file_path` | `str`, optional   | Custom path to save mask info. If `None`: <br>`"{save_dir}/param_S/top{top_ratio}"` (mode='S') <br>`"{save_dir}/param_grad_2/top{top_ratio}"` otherwise |

**Notes:**

* Hooks are only registered on parameters with `requires_grad=True`.
* For distributed models (`DataParallel` or `DistributedDataParallel`), `model.module` is used for hook registration.
* Not tested with DeepSpeed Zero-3.

**Example:**

```python
mask_grads(
    model=model,
    mode="S",
    save_dir="./mask_output",
    top_ratio=0.001
)
```

---

## Partial Code Sources

* [TRACE](https://github.com/BeyonderXX/TRACE)
* [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/)

---

## Authors

* TAP
* Lx Wang
* Hn Zhang
