"""
重新生成 GOT-10k_Val_000156 颜色阈值法的轨迹图。
以视频第一帧为背景，绘制红色预测轨迹和绿色真实轨迹。
保存为另一张图片，不覆盖原文件。
"""
import os
import sys
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(family='SimHei', size=12)
plt.rcParams['axes.unicode_minus'] = False

seq_name = "GOT-10k_Val_000156"
method_name = "color"
input_dir = f"D:/experiment/SingleTrack_Project/data/datasets/GOT_test/{seq_name}"
save_dir = f"D:/experiment/SingleTrack_Project/results/traditional/{seq_name}/{method_name}"

# 读取第一帧作为背景
img_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
first_frame = cv2.imread(os.path.join(input_dir, img_files[0]))
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

# 读取GT
gt_boxes = []
with open(os.path.join(input_dir, 'groundtruth.txt'), 'r') as f:
    for line in f:
        p = line.strip().split(',')
        x, y, w, h = int(float(p[0])), int(float(p[1])), int(float(p[2])), int(float(p[3]))
        gt_boxes.append((x, y, w, h))

# 读取预测
pred_boxes = []
with open(os.path.join(save_dir, 'trajectory.csv'), 'rb') as f:
    text = f.read().decode('utf-8-sig')
for line in text.strip().split('\n')[1:]:
    p = line.strip().split(',')
    x, y, w, h = int(float(p[1])), int(float(p[2])), int(float(p[3])), int(float(p[4]))
    pred_boxes.append((x, y, w, h))

# 计算中心点
gt_centers = [(x+w//2, y+h//2) for x,y,w,h in gt_boxes]
pred_centers = [(x+w//2, y+h//2) for x,y,w,h in pred_boxes]

print(f"GT: {len(gt_centers)}帧, 预测: {len(pred_centers)}帧")
print(f"GT X范围: {min(c[0] for c in gt_centers)}~{max(c[0] for c in gt_centers)}")
print(f"GT Y范围: {min(c[1] for c in gt_centers)}~{max(c[1] for c in gt_centers)}")
print(f"预测 X范围: {min(c[0] for c in pred_centers)}~{max(c[0] for c in pred_centers)}")
print(f"预测 Y范围: {min(c[1] for c in pred_centers)}~{max(c[1] for c in pred_centers)}")

# 在视频首帧上绘制轨迹
fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(first_frame_rgb)

# 真实轨迹（绿色）
gt_xs = [c[0] for c in gt_centers]
gt_ys = [c[1] for c in gt_centers]
ax.plot(gt_xs, gt_ys, 'g-', linewidth=2, label='真实轨迹', zorder=4)

# 预测轨迹（红色）
pred_xs = [c[0] for c in pred_centers]
pred_ys = [c[1] for c in pred_centers]
ax.plot(pred_xs, pred_ys, 'r-', linewidth=2, label='预测轨迹', zorder=5)

# 起始点（蓝色大圆）
ax.scatter(gt_centers[0][0], gt_centers[0][1], color='blue', s=200, marker='o',
           zorder=10, label='起始点', edgecolors='white', linewidths=2)

ax.set_title(f'颜色阈值法 - 目标轨迹 ({seq_name})', fontproperties=FontProperties(family='SimHei', size=14))
ax.legend(prop=font)
ax.axis('off')

plt.tight_layout()
output_path = os.path.join(save_dir, f'{seq_name}_{method_name}_trajectory_v2.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: {output_path}")