# PIECE
[English](README_en.md) | [中文](README.md)
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
    <a href="None">iusses</a>
  </p>

</p> -->
 
# 目录

- [快速开始](#快速开始)
- [简单示例](#简单示例)
- [主要方法](#主要方法)
- [部分代码来源](#部分代码来源)
- [作者](#作者)
<!-- - [鸣谢](#鸣谢) -->

---

## 快速开始
### 环境安装
```
git clone https://github.com/wanglingxiang0717/PIECE.git
cd PIECE/piece
pip install -e . --no-build-isolation
```
### 使用
```python
from piece import process_mask, mask_grads

model_load 
dataset
data_collator #model, data, data_collator 初始化之后

save_dir = args.output_dir
top_ratio = 0.001
mode = "S"  #["S", "F"]
process_mask(model = model, 
              dataset = train_dataset, 
              data_collator = data_collator, 
              save_dir = save_dir, 
              mode = mode, 
              top_ratio = top_ratio,
              args = args)

optimizer_creat #优化器创建前

mask_grads(model=model, 
            mode = mode, 
            save_dir = save_dir,
            top_ratio = top_ratio)

train #训练前
```
---
## 简单示例
### ***simple_torch***
```shell
cd example
chmod +x scripts/torch_train.sh
sh scripts/torch_train.sh
```
*example/training/torch_train_main_deepspeed.py*
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
```
cd example
chmod +x scripts/trainer_train.sh
sh scripts/trainer_train.sh
```
*example/training/trainer_train_main_deepspeed.py*
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

llama_factory 目前没有做特别适配，训练逻辑主要在 *LLaMA-Factory/src/llamafactory/train/{method}/workflow.py* 文件中

以sft为例（用*example/training/workflow.py*替换源文件中的对应文件）
```python
from piece import process_mask, mask_grads #30

save_dir = training_args.output_dir #70
top_ratio = 0.001
mode = "S"
train_dataset = dataset_module["train_dataset"]
process_mask(model = model, 
              dataset = train_dataset, 
              data_collator = data_collator, 
              save_dir = save_dir, 
              mode = mode, 
              top_ratio = top_ratio,
              args = training_args,
              cpu_offload=True
              )
if training_args.local_rank != -1:
    torch.distributed.barrier()

mask_grads(model=model,   #86
            mode = mode, 
            save_dir = save_dir,
            top_ratio = top_ratio)
```

## 主要方法

### `process_mask` 方法说明

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

### 功能

处理并保存模型参数的掩码（mask），用于选择性保留重要参数或进行梯度掩码计算。

### 参数说明

| 参数              | 类型                                         | 说明                                                                                           |
| --------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `model`         | `torch.nn.Module`                          | 需要进行掩码处理的模型。                                                                                 |
| `dataset`       | `torch.utils.data.Dataset`                 | 用于梯度计算的数据集。                                                                                  |
| `data_collator` | `any`                                  | 数据批处理函数（例如 Hugging Face 的 `DataCollatorForSeq2Seq`）。                                         |
| `save_dir`      | `str`                                      | 掩码文件的保存目录，包括中间文件和最终结果。                                                                       |
| `mode`          | `str`                                      | 掩码模式，可选 `'F'`（Fisher）或 `'S'`（二阶归一化 Second-order normalization）。                              |
| `top_ratio`     | `float`                                    | 保留的参数比例（例如 `0.001` 表示保留 0.1% 最重要参数）。                                                         |
| `args`          | `argparse.Namespace`                       | 包含配置的参数对象，例如 `local_rank`、分布式训练相关设置。                                                         |
| `singleGPU`     | `bool`, 可选，默认 `True`                       | 是否在单 GPU 上计算梯度，推荐启用以保证安全和一致性。                                                                |
| `cpu_offload`   | `bool`, 可选，默认 `False`                      | 是否将梯度计算卸载到 CPU，以节省 GPU 内存（会增加计算时间）。                                                          |
| `loss_function` | `Callable`, 可选，默认 `standard_loss_function` | 用户可自定义损失函数，签名如下：<br>`loss = loss_function(model, batch, device, args)`<br>可用于自定义训练目标或模型前向计算。 |

---

### 用法示例

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

### `mask_grads` 方法说明

```python
def mask_grads(
        model: torch.nn.Module,
        mode: str,
        save_dir: str,
        top_ratio: float,
        mask_file_path: str = None
    ):
```

### 功能

在模型参数上注册梯度钩子（gradient hook），用于选择性掩码梯度，以便后续处理或保存掩码信息。

### 参数说明

| 参数               | 类型                | 说明                                                                                                                                        |
| ---------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | `torch.nn.Module` | 需要注册梯度钩子的模型。                                                                                                                              |
| `mode`           | `str`             | 掩码模式，可选 `'F'`（Fisher）或 `'S'`（二阶归一化 Second-order normalization）。                                                                           |
| `save_dir`       | `str`             | 掩码信息保存目录，包括中间文件和最终结果。                                                                                                                     |
| `top_ratio`      | `float`           | 保留的参数比例（例如 `0.001` 表示保留 0.1% 最重要参数）。                                                                                                      |
| `mask_file_path` | `str`, 可选         | 自定义掩码信息保存路径。如果为 `None`，默认路径如下：<br>- `"{save_dir}/param_S/top{top_ratio}"`（mode='S'）<br>- `"{save_dir}/param_grad_2/top{top_ratio}"`（F） |

---

### 注意事项

* 钩子只注册在 `requires_grad=True` 的参数上。
* 对于分布式模型（`torch.nn.DataParallel` 或 `DistributedDataParallel`），会使用底层 `model.module` 进行钩子注册。
* 目前没有对deepspeed zero-3方法做测试！
---

### 用法示例

```python
mask_grads(
    model=model,
    mode="S",
    save_dir="./mask_output",
    top_ratio=0.001
)
```

---
### 部分代码来源

- [TRACE](https://github.com/BeyonderXX/TRACE)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/)

---
### 作者
* TAP
* Lx Wang
* Hn Zhang

<!-- 
### 鸣谢 -->

<!-- links -->
<!-- [your-project-path]:shaojintian/Best_README_template
[contributors-shield]: https://img.shields.io/github/contributors/shaojintian/Best_README_template.svg?style=flat-square
[contributors-url]: https://github.com/shaojintian/Best_README_template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shaojintian/Best_README_template.svg?style=flat-square
[forks-url]: https://github.com/shaojintian/Best_README_template/network/members
[stars-shield]: https://img.shields.io/github/stars/shaojintian/Best_README_template.svg?style=flat-square
[stars-url]: https://github.com/shaojintian/Best_README_template/stargazers
[issues-shield]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/shaojintian/Best_README_template.svg
[license-shield]: https://img.shields.io/github/license/shaojintian/Best_README_template.svg?style=flat-square
[license-url]: https://github.com/shaojintian/Best_README_template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/shaojintian -->




