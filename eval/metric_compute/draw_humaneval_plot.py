import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.figsize'] = (15, 9)
plt.rcParams['font.size'] = 28

markers = [
    'o', 's', '^', 'D',   # 色系1
    'v', '>',              # 色系2
    '<', 'p',              # 色系3
    '*', 'X',              # 色系4
    'P', 'h'               # 色系5（最后一个突出颜色用新的 marker）
]

colors = [
    "#c4c2e2",  # 浅蓝紫，增强可视性
    "#97b1e0",  # 浅蓝
    "#6098d8",  # 中蓝
    "#3a7bbd",  # 深蓝

    # 色系2：青色系
    "#93c1c9",  # 浅青
    "#5ca3ad",  # 中青

    # # 色系3：绿色系
    # "#b6d5a7",  # 浅绿
    # "#689852",  # 中绿

    # # 色系4：橙色系
    # "#f4c59e",  # 浅橙
    # "#e08856",  # 中橙

    # 色系5：红紫系（突出）
    "#d77ea0",  # 浅红紫
    "#bf1f6d"   # 突出红紫
]

result_file_path = "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/humaneval_results_matrix_final_llama3-8b.csv"
df = pd.read_csv(result_file_path)

headers = df.columns.tolist()[1:7] + df.columns.tolist()[-2:]

x = range(1, len(df) + 1)

for i, target in enumerate(headers):
    if target.startswith("Fisher"):
        label = "SPTC-F"
    elif target.startswith("ours"):
        label = "SPTC-N"
    elif target.startswith("lora"):
        label = "SeqLoRA"
    elif target.startswith("Norm"):
        label = "LayerNorm"
    elif target.startswith("Full"):
        label = "SeqFT"
    elif target == "replay":
        label = "Replay"
    elif target == "replay-online":
        label = "Replay-online"
    else:
        label = target

    # 取得原始 y 值
    y = df[target].values.tolist()
    x_with_init = [0] + list(x)
    y_with_init = [0.32] + y

    plt.plot(x_with_init, y_with_init,
             marker=markers[i],
             label=label,
             color=colors[i],
             linewidth=3,
             markersize=12)

fig = plt.gcf()
ax = plt.gca()

# 获取当前绘图区在 figure 坐标系下的位置 [left, bottom, width, height]
box = ax.get_position()  

# 给 legend 腾出上方空间：把 Axes 向下压一点（调整比例自己微调）
top_margin_rel = 0.12 
new_ax_height = box.height * (1.0 - top_margin_rel)
ax.set_position([box.x0, box.y0, box.width, new_ax_height])

# 计算 legend 的 bbox_to_anchor（figure 坐标系）
legend_left = box.x0
legend_width = box.width
legend_bottom = box.y0 + new_ax_height  # legend 的“底部”应在 Axes 顶部
legend_height = box.height * top_margin_rel  # 给 legend 一个小高度（相对值）

# 使用 fig.legend：在 figure 空间精确放置，并用 mode='expand' 使 legend 外框填满我们指定的宽度
ncols = max(1, int(len(headers) / 2))  # 确保为正整数

legend = fig.legend(
    loc='lower left',
    bbox_to_anchor=(legend_left, legend_bottom, legend_width, legend_height),
    bbox_transform=fig.transFigure,
    ncol=ncols,
    fontsize=25,
    frameon=False,
    mode='expand',          # 让 legend 区域横向填满 bbox 的宽度
    columnspacing=1.0,      # 列间距（适当调节）
    handletextpad=0.6,      # marker 与文字间距
    borderaxespad=0.0
)

# plt.tick_params(axis='y', length=0)
# plt.xlim(left=0, right=len(df) + 0.2)
# plt.xticks(range(1, len(df) + 1))
# plt.tick_params(axis='x')

ax = plt.gca()
ax.spines['left'].set_position(('data', 0))   
ax.spines['bottom'].set_position(('data', 0)) 
plt.tick_params(axis='x')
plt.tick_params(axis='y', length=0)

plt.xlim(0, len(df) + 0.2)
plt.ylim(bottom=0)   
plt.xticks(range(1, len(df) + 1))

# plt.xlabel("Epoch")
plt.xlabel("Task Num", labelpad=10, fontsize=36, fontweight='medium')
plt.ylabel("Pass@K")

plt.grid(True, which='major', axis='y', color='#D3D3D3',
         linestyle=(0, (5, 5)), linewidth=3, zorder=0)

for spine in ['top', 'left', 'right', 'bottom']:
    ax.spines[spine].set_linewidth(3)
ax.spines['bottom'].set_zorder(30)

plt.savefig(
    "/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/test_humaneval_new_1014_test1_new.jpg",
    dpi=300,
    bbox_inches='tight',
)


# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),fontsize=25, ncol=len(headers) / 2)
# plt.legend(
#     loc='lower center',
#     bbox_to_anchor=(0.5, 1.05),  # 控制高度
#     fontsize=25,
#     ncol=int(len(headers) / 2),
#     frameon=False,
#     borderaxespad=0.0,
#     columnspacing=0.8,           # 调整列间距
#     handletextpad=0.4            # 调整符号与文字间距
# )
# plt.tick_params(axis='y', length=0)
# plt.tick_params(axis='x')
# plt.xlabel("Step")
# plt.ylabel("Code Score")

# # 网格优化
# plt.grid(True, which='major', axis='y', color='#D3D3D3', linestyle=(0, (5, 5)),
#          linewidth=3, zorder=0)
# plt.tight_layout()
# ax = plt.gca()

# ax.spines['top'].set_linewidth(3)
# ax.spines['left'].set_linewidth(3)
# ax.spines['right'].set_linewidth(3)
# ax.spines['bottom'].set_linewidth(3)
# ax.spines['bottom'].set_zorder(30)

# plt.savefig("/data/TAP/wanglingxiang/wanglingxiang/model/model_save_new/0924/test_humaneval_new_1014.png", dpi=300)
# plt.show()
