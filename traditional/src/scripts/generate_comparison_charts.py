"""
生成四种传统跟踪方法的综合对比柱状图，统一风格。
颜色阈值法使用更新后的数据。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'd:/experiment/SingleTrack_Project/results/traditional'
os.makedirs(output_dir, exist_ok=True)

# 统一颜色
bar_color = '#6E8FB2'
label_color = '#333333'

# 方法名称
methods = ['帧差法', '背景减除法', '颜色阈值法', '边缘轮廓法']
x_pos = range(len(methods))


def make_bar_chart(data, title, ylabel, fmt_str, output_name, y_lim=None):
    """生成统一风格的柱状图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(x_pos, data, width=0.5, color=bar_color, edgecolor='white', linewidth=0.5)

    # 柱子顶部数值标签
    for bar, val in zip(bars, data):
        label_text = fmt_str.format(val)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(data) * 0.02,
                label_text, ha='center', va='bottom', fontsize=11,
                color=label_color, fontweight='bold')

    # 轴标签和标题
    ax.set_xlabel('方法', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # X轴刻度
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)

    # 去除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 横向虚线网格
    ax.yaxis.grid(True, linestyle='--', color='#CCCCCC', linewidth=0.5)
    ax.set_axisbelow(True)

    # Y轴范围
    if y_lim:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim(0, max(data) * 1.18)

    plt.tight_layout()
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {output_path}')


# ========== 图1：平均IoU对比 ==========
make_bar_chart(
    data=[0.2894, 0.3052, 0.2477, 0.2842],
    title='四种方法平均IoU对比',
    ylabel='平均IoU',
    fmt_str='{:.4f}',
    output_name='comparison_iou.png',
    y_lim=(0, 0.5)
)

# ========== 图2：平均CLE对比 ==========
make_bar_chart(
    data=[254.63, 191.59, 221.89, 309.69],
    title='四种方法平均中心位置误差对比',
    ylabel='平均CLE（像素）',
    fmt_str='{:.2f}',
    output_name='comparison_cle.png',
    y_lim=(0, 380)
)

# ========== 图3：成功帧比例对比 ==========
make_bar_chart(
    data=[0.2076, 0.1998, 0.0726, 0.2114],
    title='四种方法跟踪成功帧比例对比',
    ylabel='成功帧比例',
    fmt_str='{:.4f}',
    output_name='comparison_success_rate.png',
    y_lim=(0, 0.35)
)

# ========== 图4：平均FPS对比 ==========
make_bar_chart(
    data=[434.86, 139.10, 323.5, 139.91],
    title='四种方法平均FPS对比',
    ylabel='FPS',
    fmt_str='{:.2f}',
    output_name='comparison_fps.png',
    y_lim=(0, 520)
)

print('\n全部完成！')