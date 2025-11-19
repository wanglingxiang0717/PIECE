import os
import json
import pandas as pd
from tqdm import tqdm

# 导入评测函数
from eval_20Minuten import eval_20Minuten
from eval_CStance import eval_CStance
from eval_FOMC import eval_FOMC
from eval_MeetingBank import eval_MeetingBank
from eval_NumGLUE_cm import eval_NumGLUE_cm
from eval_NumGLUE_ds import eval_NumGLUE_ds
from eval_Py150 import eval_Py150
from eval_ScienceQA import eval_ScienceQA

# 数据集对应评测函数
eval_func_dict = {
    "20Minuten": eval_20Minuten,
    "C-STANCE": eval_CStance,
    "FOMC": eval_FOMC,
    "MeetingBank": eval_MeetingBank,
    "NumGLUE-cm": eval_NumGLUE_cm,
    "NumGLUE-ds": eval_NumGLUE_ds,
    "Py150": eval_Py150,
    "ScienceQA": eval_ScienceQA
}

target_list = [
    # "base",
    # "EWC",
    "GEM",
    # "O-LoRA",
    # "LwF",
    # "lora",
    # "MIGU",
    # "Norm",
    # "replay",
    # "replay_online"
    ]
datasets = ["C-STANCE", "FOMC", "MeetingBank", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "Py150"]

for target in target_list:
    for dataset in tqdm(datasets):
        index = datasets.index(dataset)
        # model_path_base = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/{target}/{dataset}/"
        # model_path_base = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline_14B/model_result/{target}_result_qwen3_14b"

        # for i in range(5):
        model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline_14B/model_result/{target}_result_qwen3_14b/{dataset}/1"
        eval_list = []

        # 遍历该模型测试过的所有数据集
        for test_data in datasets[:index + 1]:
            file_save_path = f"{model_path}/{test_data}_multibatch_result.json"
            if not os.path.exists(file_save_path):
                print(f"Warning: {file_save_path} not found, skipping.")
                continue

            # 获取对应评测函数
            eval_func = eval_func_dict.get(test_data)
            if eval_func is None:
                print(f"Warning: No evaluation function for {test_data}")
                continue

            # 进行评测
            res = eval_func(file_save_path)
            if "accuracy" in res:
                score = res["accuracy"]
            elif "bleu-4" in res and "rouge-L" in res:
                score = (res["bleu-4"] + res["rouge-L"]) / 2
            elif "similarity" in res:
                score = res["similarity"] * 0.01
            else:
                score = None

            eval_list.append({
                "dataset": test_data,
                "score": score
            })

        # 保存单个模型的 CSV
        if eval_list:
            df = pd.DataFrame(eval_list)
            csv_save_path = os.path.join(model_path, f"evaluation_result.csv")
            df.to_csv(csv_save_path, index=False)
            print(f"Saved evaluation for {model_path} to {csv_save_path}")
