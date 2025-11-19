# import os
# import json
# import pandas as pd
# from tqdm import tqdm

# # 导入你的评测函数
# from eval_20Minuten import eval_20Minuten
# from eval_CStance import eval_CStance
# from eval_FOMC import eval_FOMC
# from eval_MeetingBank import eval_MeetingBank
# from eval_NumGLUE_cm import eval_NumGLUE_cm
# from eval_NumGLUE_ds import eval_NumGLUE_ds
# from eval_Py150 import eval_Py150
# from eval_ScienceQA import eval_ScienceQA

# # 数据集与对应的评测函数
# eval_func_dict = {
#     "20Minuten": eval_20Minuten,
#     "C-STANCE": eval_CStance,
#     "FOMC": eval_FOMC,
#     "MeetingBank": eval_MeetingBank,
#     "NumGLUE-cm": eval_NumGLUE_cm,
#     "NumGLUE-ds": eval_NumGLUE_ds,
#     "Py150": eval_Py150,
#     "ScienceQA": eval_ScienceQA
# }

# target_list = ["Full", "Fisher", "ours"]
# datasets = ["C-STANCE", "FOMC", "MeetingBank", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "Py150"]

# # 用于存储最终CSV数据
# all_results = []

# for target in target_list:
#     for dataset in tqdm(datasets):
#         index = datasets.index(dataset)
#         name_chain = '_'.join(datasets[: index + 1])
#         model_path_base = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/{target}_{name_chain}_epoch5_Llama3Exp_0.001"

#         for i in range(5):
#             model_path = f"{model_path_base}/{i + 1}"
#             row = {"target": target, "model_index": i + 1, "model_path": model_path}

#             # 遍历模型对应的所有测试数据集
#             for test_data in datasets[:index + 1]:
#                 file_save_path = f"{model_path}/{test_data}_multibatch_result.json"
#                 if not os.path.exists(file_save_path):
#                     print(f"Warning: {file_save_path} not found, skipping.")
#                     continue

#                 # 获取对应评测函数
#                 eval_func = eval_func_dict.get(test_data)
#                 if eval_func is None:
#                     print(f"Warning: No evaluation function for {test_data}")
#                     continue

#                 # 进行评测
#                 eval_res = eval_func(file_save_path)

#                 # 根据指标决定存储值
#                 if "accuracy" in eval_res:
#                     row[test_data] = eval_res["accuracy"]
#                 elif "bleu_4" in eval_res and "rouge-L" in eval_res:
#                     row[test_data] = (eval_res["bleu-4"] + eval_res["rouge-L"]) / 2
#                 elif "similarity" in eval_res:
#                     row[test_data] = eval_res["similarity"] * 0.01
#                 else:
#                     # 如果没有对应指标，可以选择存NaN
#                     row[test_data] = None

#             all_results.append(row)

# # 保存为CSV
# df = pd.DataFrame(all_results)
# csv_save_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/analyze_result_files/0924/eval_results_summary.csv"
# df.to_csv(csv_save_path, index=False)
# print(f"Saved evaluation results to {csv_save_path}")


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
            # "Full", 
            "Fisher", 
            # "ours"
            ]
datasets = ["C-STANCE", "FOMC", "MeetingBank", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten", "Py150"]
# datasets = ["Py150"]

for target in target_list:
    for dataset in tqdm(datasets):
        index = datasets.index(dataset)
        name_chain = '_'.join(datasets[: index + 1])
        model_path_base = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/1022/{target}_{name_chain}_epoch1_Llama3Exp_0.01"

        # for i in range(5):
        model_path = f"{model_path_base}/{1}"
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
