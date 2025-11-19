from transformers import LlamaForCausalLM, AutoTokenizer
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

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

def write_json_or_jsonl(file_path, data, encoding='utf-8'):
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'w', encoding=encoding) as f_w:
            json.dump(data, f_w, ensure_ascii=False, indent=2)
    elif txt == 'jsonl':
        with open(file_path, 'w', encoding=encoding) as f_w:
            for line in data:
                json_line = json.dumps(line)
                f_w.write(json_line + '\n')
    else:
        print("file_path not exisits")
        
def tokenize(tokenizer, sentence, cutoff_len, add_bos_token=True, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        sentence,
        truncation=True,
        max_length=cutoff_len,
        add_special_tokens=False,
        padding=False,
        return_tensors=None,
    )

    if (
            len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if (
            len(result["input_ids"]) < cutoff_len
            and add_bos_token
    ):
        result["input_ids"] = [tokenizer.bos_token_id] + result["input_ids"]
        result["attention_mask"] = [1] + result["attention_mask"]

    result["labels"] = result["input_ids"].copy()

    return result

model_path = "/data2/TAP/model/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path)

target_list = [
    "20Minuten",
    "C-STANCE",
    "FOMC",
    "MeetingBank",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "Py150",
    "ScienceQA",
]

# target_list = [
#     "arabic_qa",
#     "assamese_in",
#     "bangla_bd",
#     "bangla_in",
#     "hindi_in",
#     "nepali_np",
#     "turkish_tr",
#     "english_bd",
#     "english_qa",
# ]

detection_list = {}
for target in tqdm(target_list):
    # count = 0
    max_input_length = 0
    max_answer_length = 0
    input_length = []
    answer_length = []
    file_path_dir = f"/home/TAP/TRACE-Benchmark/LLM-CL-Benchmark_1000/{target}"
    for file_name in os.listdir(file_path_dir):
        if file_name.endswith('.json'):
            data = read_json_or_jsonl(os.path.join(file_path_dir, file_name))
            for item in data:
                instruction = item["prompt"]
                label = item['answer']
                tokenized_instruction = tokenize(tokenizer, instruction, 2048000, add_bos_token=True, add_eos_token=False)["input_ids"]
                tokenized_label = tokenize(tokenizer, label, 2048000, add_bos_token=False, add_eos_token=True)["input_ids"]
                if len(tokenized_instruction) > max_input_length:
                    max_input_length = len(tokenized_instruction)
                if len(tokenized_label) > max_answer_length:
                    max_answer_length = len(tokenized_label)
                input_length.append(len(tokenized_instruction))
                answer_length.append(len(tokenized_label))
    avg_input_length = int(sum(input_length) / len(input_length))
    avg_answer_length = int(sum(answer_length) / len(answer_length))
    detection_list[target] = [max_input_length, max_answer_length, avg_input_length, avg_answer_length]
                
    # for i in range(len(data)):
    #     item = data[i]
    #     instruction = item['prompt']
    #     label = item['answer']
    #     tokenized_instruction = tokenize(tokenizer, instruction, 2048, add_bos_token=True, add_eos_token=False)
    #     tokenized_label = tokenize(tokenizer, label, 2048, add_bos_token=False, add_eos_token=True)
    #     tokenize_source = BatchEncoding({
    #         "input_ids": tokenized_instruction["input_ids"] + tokenized_label["input_ids"],
    #         "attention_mask": tokenized_instruction["attention_mask"] + tokenized_label["attention_mask"],
    #         "labels": tokenized_instruction["labels"] + tokenized_label["labels"],
    #     })
    #     # tokenized_label = self.tokenize(label, limit_len, add_bos_token=False, add_eos_token=True)
    #     tokenize_source_test = tokenize(tokenizer, instruction + label, 2048, add_bos_token=True, add_eos_token=True)
    #     # stop = input(tokenize_source)
    #     # stop = input(tokenize_source_test)
    #     if tokenize_source_test['input_ids'] != tokenize_source['input_ids']:
    #         count += 1
    # # if target not in detection_list:
    # detection_list[target] = count

print(detection_list)
                

