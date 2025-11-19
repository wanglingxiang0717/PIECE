import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 38

# colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52"]
# colors = [
#     "#2E86AB",  # 深蓝，稳重、优雅
#     "#F6AA1C",  # 金黄，醒目但柔和
#     # "#54B368",  # 草绿，清新自然
#     "#9C88FF",  # 浅紫，柔和优雅
#     "#6F4CFF"   # 亮紫，更突出，作为重点
# ]

colors = [
"#2E86AB",  # 1 (原颜色)
# "#F6AA1C",  # 2 (原颜色)
"#54B368",  # 3 (原颜色)
# "#9C88FF",  # 4 (原颜色)
# "#6F4CFF",  # 5 (原颜色)
"#1A5E78",  # 6 (色系A-深蓝)
"#E74C3C",  # 7 (色系B-红色)
"#3D9FC7",  # 8 (色系A-中蓝)
"#EC7063",  # 9 (色系B-浅红)
"#64B5F6",  # 10 (色系A-浅蓝)
"#F1948A"   # 11 (色系B-粉红)
]

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

# target_list = [
#     "Full",
#     "lora",
#     "Fisher",
#     "ours",
# ]
# target_use = [
#     "Full_parameters",
#     "LoRA",
#     "Ours Fisher",
#     "Ours NT"
# ]

target_list = [
    # "Full",
    # "EWC",
    # "GEM",
    # "lora",
    # "O-LoRA",
    # "LwF",
    # "replay",
    # "replay_online",
    # "MIGU"
    # "Norm"
    # "Fisher"
    "ours"
    # "Fisher_0001",
    # "ours_0001",
    # "Fisher_001",
    # "ours_001",
    # "Fisher_005",
    # "ours_005",
]

model_path_list = {
    # "0001": ['0924', 0.001],
    "0001": ['1013', 0.001],
    "001": ['1001', 0.01],
    "005": ['1002', 0.05],
}

df_target_list = []

step_context = {}
step_length = None

for target in target_list:
    df_mean_list = []
    for dataset in datasets_list[:-1]: 
        index = datasets_list.index(dataset)
        name_chain = '_'.join(datasets_list[: index + 1])
        
        for i in range(5):
            if target in ["Fisher", "ours"]:
                model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/1020/{target}_{name_chain}_epoch5_Llama3Exp_0.001/{i+1}/evaluation_result.csv"
            elif target == "Full":
                model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/2B_model_result/model_result/base_result_gemma-2/{dataset}/{i+1}/evaluation_result.csv"
            else:
                model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/2B_model_result/model_result/{target}_result_gemma-2/{dataset}/{i + 1}/evaluation_result.csv"
            if not os.path.exists(model_path):
                print(f"Warning: {model_path} not found, skipping.")
                continue
            
            df = pd.read_csv(model_path)

            forget_scores = []

            for data_detection in df['dataset']:
                if data_detection == datasets_list[-1]:
                    continue

                idx_old = datasets_list.index(data_detection)
                name_chain_old = '_'.join(datasets_list[: idx_old + 1])
                if target in ["Fisher", "ours"]:
                    old_model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/1020/{target}_{name_chain_old}_epoch5_Llama3Exp_0.001/5/evaluation_result.csv"
                elif target == "Full":
                    old_model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/2B_model_result/model_result/base_result_gemma-2/{data_detection}/5/evaluation_result.csv"
                else:
                    old_model_path = f"/data/TAP/wanglingxiang/wanglingxiang/model/2B_model_result/model_result/{target}_result_gemma-2/{data_detection}/5/evaluation_result.csv"
                if not os.path.exists(old_model_path):
                    print(f"Warning: {old_model_path} not found, skipping.")
                    continue

                df_detection = pd.read_csv(old_model_path)
                detection_score = df_detection.loc[df_detection['dataset'] == data_detection, 'score'].values[0]

                current_score = df.loc[df['dataset'] == data_detection, 'score'].values[0]
                forget = current_score - detection_score
                forget_scores.append(forget)

            if forget_scores:
                mean_forget = sum(forget_scores) / len(forget_scores)
            else:
                mean_forget = None
            df_mean_list.append(mean_forget)
    df_target_list.append(df_mean_list)

stop = input(df_target_list)
df_avg = [df_list[-1] for df_list in df_target_list]
stop = input(df_avg)

def apply_ema(data, alpha=0.01):
    ema_values = []
    ema = data[0]  
    for value in data:
        ema = alpha * value + (1 - alpha) * ema  # 计算 EMA
        ema_values.append(ema)
    return ema_values

y_value = df_target_list

# y_value = [apply_ema(y_value[i], alpha=0.5) for i in range(len(y_value))]

# point_data = [[[int(df_target_list[k].iloc[i, 0]), df_target_list[k].iloc[i, 1]] for i in range(len(df_target_list[k]))] for k in range(len(df_target_list))]
x_value = [list(range(1, len(y_value[i]) + 1)) for i in range(len(y_value))]

for i in range(len(x_value)):
    plt.plot(x_value[i], y_value[i], label=target_list[i], color=colors[i], linewidth=3)

# for i in range(len(target_list)):
#     for k in range(min([len(step_context[target]) for target in target_list])):
#         y_base = max([step_context[target][k] for target in target_list])
#         plt.text(step_length * (k + 1), y_base + 0.2 + i * 0.2, f"{step_context[target_list[i]][k]:.4f}", fontsize=17, color=colors[i], fontweight='bold')

plt.legend(loc="best", fontsize=25)

# ax = plt.gca()
# y_min, y_max = ax.get_ylim()

# y_ticks = np.arange(np.floor(y_min*10)/10, np.ceil(y_max*10)/10 + 0.05, 0.05)
# ax.set_yticks(y_ticks)

plt.tick_params(axis='y', length=0)
# plt.tick_params(axis='y')
plt.tick_params(axis='x')

# plt.legend()
plt.title('Average Forget BWT')
# plt.title('Varying degrees of learning.(all epoch)')
# plt.title('Varying degrees of forgetting.(all epoch)')

plt.xlabel("Step")
# plt.ylabel("Loss of the previous task(eval dataset).")
plt.ylabel("Score")

plt.grid(True, which='major', axis='y', color='#D3D3D3', linestyle=(0, (5, 5)),
         linewidth=3, zorder=0)

ax = plt.gca()

ax.spines['top'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['bottom'].set_zorder(30)

plt.show()
plt.savefig("/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/analyze_result_files/1009/BWT_1009_withoutpy150.png")
# plt.savefig("/data2/TAP/model_con/analyze_pic/test_new_0924_all.png")
