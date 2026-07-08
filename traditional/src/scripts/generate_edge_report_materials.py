"""
生成边缘轮廓法课设报告所需全部材料和图表。
包括：
- 5关键帧跟踪结果
- 中间处理流程4子图
- 误差曲线+轨迹图
- 所有序列指标统计
"""
import os
import sys
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')

import cv2
import numpy as np
import csv
import time
from PIL import Image, ImageDraw, ImageFont
from models.traditional.edge_contour_tracker import EdgeContourTracker
from utils.evaluation import Evaluation

seq_name = "GOT-10k_Train_000424"
input_dir = f"d:/experiment/SingleTrack_Project/data/datasets/GOT_test/{seq_name}"
save_dir = f"d:/experiment/SingleTrack_Project/results/traditional/{seq_name}/edge_contour"
edge_dir = f"d:/experiment/SingleTrack_Project/results/traditional/{seq_name}/edge"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(edge_dir, exist_ok=True)


def put_label_cv2(img, text, pos=(10, 30), color=(0, 255, 0)):
    from PIL import Image, ImageDraw, ImageFont
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]
    fnt = None
    for fp in font_paths:
        if os.path.exists(fp):
            fnt = ImageFont.truetype(fp, 28)
            break
    if fnt is None:
        fnt = ImageFont.load_default()
    draw.text(pos, text, fill=color, font=fnt)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ========== 加载序列 ==========
frames = []
img_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))])
for f in img_files:
    img = cv2.imread(os.path.join(input_dir, f))
    if img is not None:
        frames.append(img)

gt = []
with open(os.path.join(input_dir, 'groundtruth.txt'), 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        gt.append((int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))))

print(f"加载 {len(frames)} 帧")

# ========== 运行跟踪器 ==========
tracker = EdgeContourTracker()
tracker.init(frames[0], gt[0])
trajectory = [gt[0]]
for i in range(1, len(frames)):
    trajectory.append(tracker.update(frames[i]))

ev = Evaluation()
with open(os.path.join(save_dir, 'trajectory.csv'), 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['帧号','x','y','w','h','IoU','CLE'])
    for i, (tx,ty,tw,th) in enumerate(trajectory):
        iou = ev.calculate_iou((tx,ty,tw,th), gt[i])
        cle = ev.calculate_center_error((tx,ty,tw,th), gt[i])
        w.writerow([i, tx, ty, tw, th, f'{iou:.4f}', f'{cle:.4f}'])

# ========== 生成中间处理图 ==========
print("\n--- 生成中间处理图 ---")
test_frame = frames[0].copy()
x0, y0, w0, h0 = gt[0]
gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)

# 1. 灰度图
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
gray_bgr = put_label_cv2(gray_bgr, "灰度图")
cv2.imwrite(os.path.join(save_dir, "grayscale.png"), gray_bgr)
print("已保存: grayscale.png")

# 2. 高斯滤波
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
blurred_bgr = put_label_cv2(blurred_bgr, "高斯滤波")
cv2.imwrite(os.path.join(save_dir, "gaussian_blur.png"), blurred_bgr)
print("已保存: gaussian_blur.png")

# 3. Canny边缘检测
canny = cv2.Canny(blurred, tracker.canny_low, tracker.canny_high)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
canny_dilated = cv2.dilate(canny, kernel, iterations=2)
canny_bgr = cv2.cvtColor(canny_dilated, cv2.COLOR_GRAY2BGR)
canny_bgr = put_label_cv2(canny_bgr, "Canny边缘检测")
cv2.imwrite(os.path.join(save_dir, "canny_edge.png"), canny_bgr)
print("已保存: canny_edge.png")

# 4. 轮廓提取
contours, _ = cv2.findContours(canny_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = test_frame.copy()
cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
cv2.rectangle(vis, (x0, y0), (x0+w0, y0+h0), (0, 255, 255), 3)
vis = put_label_cv2(vis, "轮廓提取")
cv2.imwrite(os.path.join(save_dir, "contour_frame.png"), vis)
print(f"已保存: contour_frame.png (轮廓数: {len(contours)})")

import shutil
for fn in ['grayscale.png', 'gaussian_blur.png', 'canny_edge.png', 'contour_frame.png']:
    shutil.copy2(os.path.join(save_dir, fn), os.path.join(edge_dir, fn))

# ========== 生成5关键帧跟踪结果 ==========
print("\n--- 生成5关键帧跟踪结果 ---")
total = len(frames)
key_indices = [0, int(total*0.25), int(total*0.5), int(total*0.75), total-1]
key_labels = ["第1帧", "第25帧", "第50帧", "第75帧", "第100帧"]

for idx, label in zip(key_indices, key_labels):
    frame = frames[idx].copy()
    x, y, w, h = gt[idx]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
    cv2.putText(frame, 'GT', (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    px, py, pw, ph = trajectory[idx]
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 0, 255), 3)
    cv2.putText(frame, 'Edge', (px, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    fname = f"frame_{idx:04d}.png"
    cv2.imwrite(os.path.join(save_dir, fname), frame)
    cv2.imwrite(os.path.join(edge_dir, fname), frame)
    print(f"已保存: {fname} ({label})")

# ========== 生成误差曲线图（统一风格：单张大图，与Val_000166一致） ==========
print("\n--- 生成误差曲线图 ---")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(family='SimHei', size=12)
plt.rcParams['axes.unicode_minus'] = False

ious = []
cles = []
with open(os.path.join(save_dir, 'trajectory.csv'), 'rb') as f:
    text = f.read().decode('utf-8-sig')
for line in text.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    ious.append(float(parts[5]))
    cles.append(float(parts[6]))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Left: IoU
axes[0].plot(range(len(ious)), ious, 'b-', linewidth=1.5)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('IoU')
axes[0].set_title('IoU Curve')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 1)

# Right: CLE
axes[1].plot(range(len(cles)), cles, 'r-', linewidth=1.5)
axes[1].set_xlabel('Frame')
axes[1].set_ylabel('CLE (pixels)')
axes[1].set_title('Center Location Error')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
error_path = os.path.join(save_dir, f'{seq_name}_edge_contour_error_curves.png')
plt.savefig(error_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"已保存: error_curves.png (尺寸将比参考稍小，风格一致)")

# ========== 生成轨迹图 ==========
print("\n--- 生成轨迹图 ---")
pred_centers = [(trajectory[i][0]+trajectory[i][2]//2, trajectory[i][1]+trajectory[i][3]//2) for i in range(len(trajectory))]
gt_centers = [(gt[i][0]+gt[i][2]//2, gt[i][1]+gt[i][3]//2) for i in range(len(gt))]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([c[0] for c in gt_centers], [c[1] for c in gt_centers], 'g--', label='真实轨迹', linewidth=2)
ax.plot([c[0] for c in pred_centers], [c[1] for c in pred_centers], 'r-', label='预测轨迹', linewidth=3)
ax.scatter(pred_centers[0][0], pred_centers[0][1], color='blue', s=200, marker='o',
           zorder=10, label='起始点', edgecolors='black', linewidths=2)
ax.set_xlabel('X坐标', fontproperties=font)
ax.set_ylabel('Y坐标', fontproperties=font)
ax.set_title('边缘轮廓法 - 目标轨迹', fontproperties=font)
ax.legend(prop=font)
ax.grid(True, alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()

traj_path = os.path.join(save_dir, f'{seq_name}_edge_contour_trajectory.png')
plt.savefig(traj_path, dpi=100)
plt.close()
print("已保存: trajectory.png")

# ========== 统计全部10序列指标 ==========
print("\n--- 统计全部10序列指标 ---")
test_dir = r'd:/experiment/SingleTrack_Project/data/datasets/GOT_test'
all_metrics = []

for sname in sorted(os.listdir(test_dir)):
    d = os.path.join(test_dir, sname)
    if not os.path.isdir(d):
        continue
    sframes = []
    for f in sorted(os.listdir(d)):
        if f.endswith(('.jpg','.png')):
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                sframes.append(img)
    sgt = []
    with open(os.path.join(d, 'groundtruth.txt'), 'r') as f:
        for line in f:
            p = line.strip().split(',')
            sgt.append((int(float(p[0])),int(float(p[1])),int(float(p[2])),int(float(p[3]))))

    t = EdgeContourTracker()
    t.init(sframes[0], sgt[0])
    traj = [sgt[0]]
    for i in range(1, len(sframes)):
        traj.append(t.update(sframes[i]))

    iou_list = [ev.calculate_iou(traj[i], sgt[i]) for i in range(len(traj))]
    cle_list = [ev.calculate_center_error(traj[i], sgt[i]) for i in range(len(traj))]
    success_rate = sum(1 for iou in iou_list if iou >= 0.5) / len(iou_list)
    fps = len(sframes) / (time.time() - t.start_time + 1e-6)

    all_metrics.append({
        'seq': sname, 'mIoU': np.mean(iou_list), 'mCLE': np.mean(cle_list),
        'success': success_rate, 'fps': fps, 'lost': t.lost_count, 'maxCLE': max(cle_list)
    })
    print(f"  [{sname}] mIoU={np.mean(iou_list):.4f} mCLE={np.mean(cle_list):.2f} success={success_rate:.2%}")

# 保存metrics.csv
metrics_path = os.path.join(save_dir, 'metrics.csv')
with open(metrics_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['序列','mIoU','mCLE','成功率','FPS','丢失次数','最大CLE'])
    for m in all_metrics:
        w.writerow([m['seq'], f"{m['mIoU']:.4f}", f"{m['mCLE']:.2f}",
                    f"{m['success']:.2%}", f"{m['fps']:.1f}", m['lost'], f"{m['maxCLE']:.2f}"])

# 同步到 edge 目录
for fn in ['trajectory.csv', 'metrics.csv']:
    shutil.copy2(os.path.join(save_dir, fn), os.path.join(edge_dir, fn))
for fn in [f'frame_{i:04d}.png' for i in key_indices]:
    shutil.copy2(os.path.join(save_dir, fn), os.path.join(edge_dir, fn))
shutil.copy2(error_path, os.path.join(edge_dir, f'{seq_name}_edge_error_curves.png'))
shutil.copy2(traj_path, os.path.join(edge_dir, f'{seq_name}_edge_trajectory.png'))

print(f"\n全部完成! 所有文件保存到 {save_dir}/ 和 {edge_dir}/")