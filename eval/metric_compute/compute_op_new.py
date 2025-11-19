import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datasets_list=[
    "C-STANCE", 
    "FOMC", 
    "MeetingBank",  
    "ScienceQA",
    "NumGLUE-cm",
    "NumGLUE-ds",
    "20Minuten",
    "Py150"
    ]


target_list = [
    # "Full",
    # "EWC",
    # "GEM",
    # "lora",
    # "O-LoRA",
    # "MIGU",
    # "Norm",
    # "LwF",
    # "replay",
    # "replay_online",
    "Fisher",
    "ours"
    # "Fisher_0001",
    # "ours_0001",
    # "Fisher_001",
    # "ours_001",
    # "Fisher_005",
    # "ours_005",
]

model_path_list = {
    "0001": ['0924', 0.001],
    "001": ['1001', 0.01],
    "005": ['1002', 0.05],
}

df_target_list = []

step_context = {}
step_length = None

for target in target_list:
    if target in ["Fisher", "ours"]:
        score_list = {}
        # target_clear = target.split("_")[0]
        # detection_list = model_path_list[target.split("_")[-1]]
        for dataset in datasets_list:
            try:
                index = datasets_list.index(dataset)
                name_chain = '_'.join(datasets_list[ : index + 1])
                df = pd.read_csv(f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/1002/{target}_{name_chain}_epoch5_Llama3Exp_0.05/{5}/evaluation_result.csv")
                # df = pd.read_csv(f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/1020/{target}_{name_chain}_epoch5_Llama3Exp_0.001/{i+1}/evaluation_result.csv")
                score = df.iloc[-1]['score']
                score_list[dataset] = score
            except:
                continue
        # stop = input(score_list)
        mean_value = sum(score_list.values()) / len(score_list)
        stop = input(f"{target}:{mean_value}")
    elif target == "Full": 
        df_mean_list = []
        for dataset in datasets_list:
            try:
                index = datasets_list.index(dataset)
                name_chain = '_'.join(datasets_list[ : index + 1])
                for i in range(5):
                    df = pd.read_csv(f"/data/TAP/wanglingxiang/wanglingxiang/model/2B_model_result/model_result/base_result_gemma-2/{dataset}/{i+1}/evaluation_result.csv")
                    mean_score = df['score'].mean()
                    df_mean_list.append(mean_score)
            except:
                continue
        df_target_list.append(df_mean_list)
        
    else: 
        df_mean_list = []
        for dataset in datasets_list:
            try:
                index = datasets_list.index(dataset)
                name_chain = '_'.join(datasets_list[ : index + 1])
                for i in range(5):
                    df = pd.read_csv(f"/data/TAP/wanglingxiang/wanglingxiang/model/2B_model_result/model_result/{target}_result_gemma-2/{dataset}/{i + 1}/evaluation_result.csv")
                    # df = pd.read_csv(f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/{target}_{name_chain}_epoch5_Llama3Exp_0.001/{i+1}/evaluation_result.csv")
                    mean_score = df['score'].mean()
                    df_mean_list.append(mean_score)
            except:
                continue
        df_target_list.append(df_mean_list)

stop = input(df_target_list)
df_avg = [df_list[-1] for df_list in df_target_list]
stop = input(df_avg)