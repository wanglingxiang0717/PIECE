import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['figure.figsize'] = (30, 9)
plt.rcParams['font.size'] = 28

# === 定义配色：前两项深蓝，后面浅蓝 ===
colors = ["#194A8A"] * 2 + ["#B2C1E2"] * 10

label_list = [
    "SeqFT", "EWC", "GEM", "LwF", "Replay", "Replay-online",
    "SeqLoRA", "O-LoRA", "LayerNorm", "MIGU", "SPTC-F", "SPTC-N"
]

file_list = [
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/Full_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_Py150_epoch5_Llama3Exp_0.001/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/EWC/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/GEM/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/LwF/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/replay/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/replay_online/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/lora/Py150/4/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/O-LoRA/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/Norm/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/baseline/MIGU/Py150/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/Fisher_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_Py150_epoch5_Llama3Exp_0.001/5/evaluation_result.csv",
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/ours_C-STANCE_FOMC_MeetingBank_ScienceQA_NumGLUE-cm_NumGLUE-ds_20Minuten_Py150_epoch5_Llama3Exp_0.001/5/evaluation_result.csv"
]

# === 只取平均分 ===
means = [pd.read_csv(f)["score"].mean() for f in file_list]

# === 从高到低排序 ===
sorted_indices = np.argsort(means)[::-1]
means_sorted = [means[i] for i in sorted_indices]
labels_sorted = [label_list[i] for i in sorted_indices]
# colors_sorted = [colors[i] for i in sorted_indices]
colors_sorted = colors

# === 绘制横向柱状图 ===
fig, ax = plt.subplots()
y_pos = np.arange(len(labels_sorted))

bars = ax.barh(
    y_pos,
    means_sorted,
    color=colors_sorted,
    height=0.6,    # 调整柱子高度以增大间距
    zorder=3
)

# 显示数值
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 0.002, bar.get_y() + bar.get_height()/2,
            f"{means_sorted[i]:.2f}",
            va='center', ha='left', fontsize=26, color='#333333')

# === 美化 ===
ax.set_xlim([min(means_sorted) * 0.95, max(means_sorted) * 1.05])
ax.set_yticks(y_pos)

yticklabels = []
for i, lbl in enumerate(labels_sorted):
    if i < 2:
        yticklabels.append(lbl)  # 暂存文本
    else:
        yticklabels.append(lbl)

ax.set_yticklabels(yticklabels, fontsize=26)

# 加粗前两项
for i, label in enumerate(ax.get_yticklabels()):
    if i < 2:
        label.set_fontweight('bold')

ax.invert_yaxis()
# ax.set_xlim([0, max(means_sorted) * 1.1])

# 去除x轴和边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_linewidth(2.5)
ax.spines['left'].set_color("#555555")
ax.spines['left'].set_zorder(5) 
# ax.tick_params(left=True, bottom=False, labelbottom=False)
ax.tick_params(
    axis='y',
    left=True,        # 显示tick
    length=8,         # tick长度
    width=2.5,        # tick线宽
    color="#555555",  # tick颜色
)
ax.xaxis.set_visible(False)

# 去掉标题，让图更简洁
plt.tight_layout()
plt.savefig("/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/test_behavior_avg_clean.jpg", dpi=300, bbox_inches='tight')
# plt.show()
