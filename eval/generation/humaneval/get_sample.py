from human_eval.data import write_jsonl, read_problems
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
import transformers
import torch
import csv
import os
import pandas as pd
from tqdm import tqdm
import jsonlines
# from test_humaneval_one import get_result

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def filter_code(completion: str) -> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def generate_one_completion(prompt, model, tokenizer):
    # 确保模型处于评估模式
    model.eval()
    
    sequences = model.generate(
        input_ids=tokenizer(prompt, return_tensors='pt').input_ids.to(model.device),
        do_sample=True,
        top_k=10,
        temperature=0.3,
        top_p=0.95,
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    return generated_text.replace(prompt, "")

def process_get_result(base_model, device_id):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": int(device_id)},
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
            "text-generation", #指定任务
            model=model,
            tokenizer = tokenizer,
            torch_dtype=torch.float16,
            # device_map={"": 0}, # 运行设备
        )

    problems = read_problems()
    num_samples_per_task = 1
    samples = [
        dict(task_id=task_id, completion=filter_code(generate_one_completion(problems[task_id]["prompt"], model, tokenizer)))
        for task_id in tqdm(problems)
        for _ in range(num_samples_per_task)
    ]
    write_jsonl(f"{base_model}/humaneval_dosample_test.jsonl", samples)
    # del tokenizer
    # get_result(f"{base_model}/humaneval_dosample_test.jsonl")    

if __name__ == "__main__":
    model_path = "/data/TAP/wanglingxiang/wanglingxiang/model/Qwen3-14B"
    process_get_result(model_path, 7)
    # model_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/Fisher_C-STANCE_epoch5_Llama3Exp_0.001/5"
    # if not os.path.exists(os.path.join(model_path, "humaneval_dosample_test.jsonl")):
    #     process_get_result(model_path, 0)   
    
    # stop = input("finish")
    
    # target_list = [
    #     # "Full",
    #     # "Fisher",
    #     # "ours",
    # ]
    # datasets = ["C-STANCE", 
    #             "FOMC", 
    #             "MeetingBank", 
    #             "ScienceQA", 
    #             "NumGLUE-cm", 
    #             "NumGLUE-ds", 
    #             "20Minuten", 
    #             "Py150"]
    # for target in target_list:
    #     for dataset in datasets[7:]:
    #         # stop = input(dataset)
    #         # continue
    #         index = datasets.index(dataset)
    #         name_chain = '_'.join(datasets[ : index + 1])
    #         model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/1002/{target}_{name_chain}_epoch5_Llama3Exp_0.05/5"
    #         if not os.path.exists(os.path.join(model_path, "humaneval_dosample_test.jsonl")):
    #             process_get_result(model_path, 7)
    #         else:
    #             print("ok")
                
    # for dataset in datasets[7:]:
    #     # stop = input(dataset)
    #     # continue
    #     index = datasets.index(dataset)
    #     name_chain = '_'.join(datasets[ : index + 1])
    #     model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/LwF/{dataset}/5"
    #     if not os.path.exists(os.path.join(model_path, "humaneval_dosample_test.jsonl")):
    #         process_get_result(model_path, 7)
    #     else:
    #         print("ok")

    # for dataset in datasets[4:6]:
    #     # stop = input(dataset)
    #     # continue
    #     index = datasets.index(dataset)
    #     name_chain = '_'.join(datasets[ : index + 1])
    #     model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/GEM/{dataset}/5"
    #     if not os.path.exists(os.path.join(model_path, "humaneval_dosample_test.jsonl")):
    #         process_get_result(model_path, 3)
    #     else:
    #         print("ok")
    