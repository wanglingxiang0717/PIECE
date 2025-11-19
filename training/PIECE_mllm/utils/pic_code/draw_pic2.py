import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 38

# colors = [
#     "#1dbde7",  # 明亮天蓝（对比紫色，冷感强）
#     "#1de7a3",  # 薄荷绿 / 青绿（清新冷色）
#     "#1d6fe7"   # 经典蓝 / 钴蓝（比原色更深、更冷）
# ]

# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]  #区分度高颜色
# colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]  #区分度高且优雅,哈哈哈哈
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
colors = [
    # "#1f77b4",  # 蓝色系，第1个颜色（前两个颜色之一）
    "#17becf",  # 蓝色系，第2个颜色（前两个颜色之一，和 #1f77b4 区分度高）
    "#98df8a",  # 绿色系，第4个颜色（第三到第五个颜色之一）
    "#ff7f0e",  # 橙色系，第6个颜色（单独一个色系）
    "#d62728",  # 红色系，第7个颜色（单独一个色系）
]


file_list = [
        # "/data2/TAP/model/FOMC_001_test_0728_epoch15_random_allparameters_loss",
        "/data2/TAP/model/FOMC_001_test_0728_epoch15_random_allparameters_loss/result_csv/epoch_all_eval_loss01.csv",
        "/data2/TAP/model/FOMC_001_test_0728_epoch15_random_allparameters_loss/result_csv/epoch_all_eval_loss2.csv",
        "/data2/TAP/model/FOMC_001_test_0728_epoch15_random_allparameters_loss/result_csv/epoch_all_eval_loss34.csv",
        "/data2/TAP/model/FOMC_001_test_0728_epoch15_random_allparameters_loss/result_csv/epoch_all_eval_loss5678.csv"
    ]
label_list = [
    # "allparameters",
    "loss01(Highly familiar)",
    "loss2(Familiar)",
    "loss34(Slightly unfamiliar)",
    "loss5678(Unknown)"
]

# df_split = pd.read_csv("/data1/TAP/model/FOMC_001_test_0714_epoch15_random42/result_csv/epoch1_eval2.csv")
df_split_random = []
for file_name in file_list:
    # stop = input(file_name)
    df_split_random_i = pd.read_csv(file_name)
    df_split_random.append(df_split_random_i)

# point_data_split = [[int(df_split.iloc[i, 0]), df_split.iloc[i, 1]] for i in range(len(df_split))]
point_data_split_random = [[[int(df_split_random[k].iloc[i, 0]), df_split_random[k].iloc[i, 1]] for i in range(len(df_split_random[k]))] 
                           for k in range(len(df_split_random))]

# x_value_split = [point[0] for point in point_data_split]
# y_value_split = [point[1] for point in point_data_split]

x_value_split_random = [[point[0] for point in point_data_split_random[i]] for i in range(len(point_data_split_random))]
y_value_split_random =[[point[1] for point in point_data_split_random[i]] for i in range(len(point_data_split_random))]

# EMA 平滑函数
def apply_ema(data, alpha=0.01):
    ema_values = []
    ema = data[0]  
    for value in data:
        if value:
            ema = alpha * value + (1 - alpha) * ema  # 计算 EMA
            ema_values.append(ema)
        else:
            ema_values.append(None)
    return ema_values

alpha = 0.01  # 平滑系数
y_value_split_random = [apply_ema(list_point, alpha) for list_point in y_value_split_random]
# y_value_split_bad_ema = apply_ema(y_value_split_bad, alpha)

plt.figure()
# plt.plot(x_value_split, y_value_split, label='deep_first_chain_numdegreefirst', color="#d000ff", linewidth=5)
# plt.plot(x_value_split, y_value_split, label='001_random_parameters', color=colors[-1], linewidth=3)

for i in range(len(y_value_split_random)):
    # plt.plot(x_value_split_random[i], y_value_split_random[i], label=f'random_select_{i}', color=colors[i], linewidth=3)
    plt.plot(x_value_split_random[i], y_value_split_random[i], label=label_list[i], color=colors[i], linewidth=3)


# plt.axvline(x=71, ymin=0.3, ymax=0.63, color='r', linestyle='--', linewidth=3)
# plt.text(71, 0.42, 'epoch1', color='r', ha='center', va='top', fontsize=15)
# plt.axvline(x=142, ymin=0.4, ymax=0.93, color='r', linestyle='--', linewidth=3)
# plt.text(142, 0.425, 'epoch2', color='r', ha='center', va='top', fontsize=15)

plt.legend(loc="lower right", fontsize=30)
plt.tick_params(axis='y', length=0)
plt.tick_params(axis='y')
plt.tick_params(axis='x')

# plt.legend()
# plt.title('Varying degrees of learning.(15 epoch)')
plt.title('Varying degrees of forgetting.(15 epoch, all_parameters)')

plt.xlabel("Step")
plt.ylabel("Loss of the previous task(eval dataset).")
# plt.ylabel("Loss of the current task(eval dataset).")

plt.grid(True, which='major', axis='y', color='#D3D3D3', linestyle=(0, (5, 5)),
         linewidth=3, zorder=0)

ax = plt.gca()

ax.spines['top'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['bottom'].set_zorder(30)

plt.show()
plt.savefig("eval_all_loss_diff_param_protection_15_epoch_all_parameters_ema.png")