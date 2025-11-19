from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
import csv 
import os
import torch.distributed as dist
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, LlavaForConditionalGeneration
# from Collections import defaultdict

def read_json_or_jsonl(file_path):
    data = []
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'r') as f_r:
            data = json.load(f_r)
    elif txt == 'jsonl':
        with open(file_path, 'r') as f_r:
            for line in f_r:
                data.append(json.loads(line))
    else:
        data = None
    return data

def return_model_and_tokenizer(model_name, device_id):
    # model_name = "/data/wanglingxiang/model/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": int(device_id)},
    )
    return model, tokenizer
# for name, parameters in model.named_parameters():
#     stop = input(name)
#     stop = input(parameters.shape)
#     stop = input(parameters.shape[-1])

def get_conversation(question, processor):
    if question:
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}"},
                {"type": "image"},
                ],
            },
        ]
    else: 
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": f"Give a short description of the image."},
                {"type": "image"},
                ],
            },
        ]       
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt

def generation_mask(model, processor, tokenizer, target, file_dir, args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    data = read_json_or_jsonl(f"{file_dir}/{target}/train.json")
    num_samples = len(data)
    
    samples_per_rank = num_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    if rank == world_size - 1:
        end_idx = num_samples

    data_rank = data[start_idx:end_idx]     
    mean_dict = {}
    for item in tqdm(data_rank, disable=(rank != 0)):
        # batch = data_rank[i : i + batch_size] if i + batch_size <= len(data_rank) else data_rank[i:]
        questions = item['prompt']
        prompts = get_conversation(questions, processor)
        p = item["pic_path"]
        raw_images = Image.open(p).convert("RGB")
        inputs = processor(
                images=raw_images,
                text=prompts,
                return_tensors='pt',
                padding=True
            ).to(model.device)
        # stop = input(inputs['input_ids'].shape)
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
        except:
            continue

        hidden_states = outputs.hidden_states
        for layer_id, layer in enumerate(hidden_states):
            # stop = input(layer.shape)
            mean = layer.abs().mean(dim=(0, 1))
            # mean_save = mean.clone()
            if layer_id not in mean_dict:
                mean_dict[layer_id] = mean.clone()
            else:
                mean_dict[layer_id] += mean.clone()
    shape_len = None           
    for layer_id in mean_dict:
        mean_dict[layer_id] = (mean_dict[layer_id] / len(data_rank))
        if shape_len is None:
            shape_len = mean_dict[layer_id].shape[-1]
            
    return mean_dict, shape_len
    
    # top_ratio = 0.001
    # mask_dict = {}
    # shape_len = None
    # for layer_id in mean_dict:
    #     mean_dict[layer_id] = (mean_dict[layer_id] / len(data))
    #     if shape_len is None:
    #         shape_len = mean_dict[layer_id].shape[-1]
    #         # stop = input(shape_len)

    # for layer_id, mean in mean_dict.items():
    #     k = max(1, int(len(mean) * top_ratio))
    #     threshold = torch.topk(mean, k).values.min()
    #     mask = (mean >= threshold)  # bool tensor
    #     mask_dict[layer_id] = mask.nonzero(as_tuple=False).squeeze()
    
    # save_file_dir = "/data/wanglingxiang/data/test_MIGU_1014"
    # for name, param in model.named_parameters():
    #     name_list = name.split(".")
    #     # stop = input(name_list)
    #     if "layers" in name_list and param.shape[-1] == shape_len:
    #         # stop = input("yes")
    #         save_file = f"{save_file_dir}/{name}.json"
    #         position_list = []
    #         layer_id = int(name_list[2])
    #         mask_list = mask_dict[layer_id]
    #         for i in mask_list:
    #             if param.dim() == 1:
    #                 pos = [i.item()]
    #                 position_list.append(pos)
    #             else:
    #                 assert param.dim() == 2
    #                 for j in range(param.shape[0]):
    #                     pos = [j, i.item()]
    #                     position_list.append(pos)
    #         with open(save_file, "w") as f:
    #             json.dump({"positions": position_list}, f)

              
    
    # for layer_id in mask_dict:
    #     stop = input(mask_dict[layer_id].shape)
    # stop = input(mask_dict)
    # stop = input(mean_dict)