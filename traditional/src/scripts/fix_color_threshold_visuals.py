"""
修复颜色阈值法（color）的中间结果图和轨迹图。
生成 color_mask.png、morphology_frame.png、contour_frame.png、trajectory.png
"""
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageDraw, ImageFont

font = FontProperties(family='SimHei', size=12)
plt.rcParams['axes.unicode_minus'] = False

seq_name = "GOT-10k_Val_000146"
method_name = "color"
save_dir = f"D:\\experiment\\SingleTrack_Project\\results\\traditional\\{seq_name}\\{method_name}"
input_dir = f"D:\\experiment\\SingleTrack_Project\\data\\datasets\\GOT_test\\{seq_name}"
os.makedirs(save_dir, exist_ok=True)


def put_label(img_cv2, text, pos=(10, 30), color=(0, 255, 0)):
    if len(img_cv2.shape) == 2:
        img_bgr = cv2.cvtColor(img_cv2, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img_cv2.copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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


def load_got_sequence(seq_path):
    frames = []
    img_files = sorted([f for f in os.listdir(seq_path) if f.endswith(('.jpg', '.png'))])
    for f in img_files:
        img = cv2.imread(os.path.join(seq_path, f))
        if img is not None:
            frames.append(img)
    gt_file = os.path.join(seq_path, 'groundtruth.txt')
    gt = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            gt.append((x, y, w, h))
    return frames, gt


print(f"加载序列: {seq_name}")
frames, gt = load_got_sequence(input_dir)
print(f"共 {len(frames)} 帧")

if len(frames) < 2:
    print("帧数不足")
    exit(1)

test_idx = 0
test_frame = frames[test_idx].copy()
bbox = gt[test_idx]
x, y, w, h = [int(v) for v in bbox]
initial_area = w * h
prev_cx, prev_cy = x + w / 2, y + h / 2
print(f"首帧GT框: ({x}, {y}, {w}, {h}), 面积={initial_area}")

# ========== 1. 修复颜色掩码（中心区域采样） ==========
print(f"\n--- 1. 修复颜色掩码 ---")

margin_x, margin_y = int(w * 0.2), int(h * 0.2)
roi = test_frame[y + margin_y:y + h - margin_y, x + margin_x:x + w - margin_x]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

h_min = max(0, int(np.percentile(hsv_roi[:, :, 0], 5)) - 5)
h_max = min(180, int(np.percentile(hsv_roi[:, :, 0], 95)) + 5)
s_min = max(0, int(np.percentile(hsv_roi[:, :, 1], 10)) - 10)
s_max = min(255, int(np.percentile(hsv_roi[:, :, 1], 90)) + 10)
v_min = max(0, int(np.percentile(hsv_roi[:, :, 2], 10)) - 10)
v_max = min(255, int(np.percentile(hsv_roi[:, :, 2], 90)) + 10)

lower = (h_min, s_min, v_min)
upper = (h_max, s_max, v_max)
print(f"中心区域采样: roi={roi.shape[1]}x{roi.shape[0]}")
print(f"HSV范围: lower={lower}, upper={upper}")

hsv_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2HSV)
color_mask = cv2.inRange(hsv_frame, lower, upper)
color_mask = color_mask.astype(np.uint8)

nonzero_count = np.count_nonzero(color_mask)
total_pixels = color_mask.shape[0] * color_mask.shape[1]
print(f"color_mask: 非零像素={nonzero_count} ({nonzero_count/total_pixels*100:.1f}%)")

color_mask_labeled = put_label(color_mask, "颜色掩码")
cv2.imwrite(os.path.join(save_dir, "color_mask.png"), color_mask_labeled)
print(f"已保存: color_mask.png")

# ========== 2. 修复形态学处理（大核+iterations+连通域过滤） ==========
print(f"\n--- 2. 修复形态学处理 ---")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
morph = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=3)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=3)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph, connectivity=8)
min_area_cc = int(initial_area * 0.05)
print(f"连通域数量: {num_labels}, 最小面积阈值: {min_area_cc}")

filtered = np.zeros_like(morph)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area_cc:
        filtered[labels == i] = 255

morph = filtered
nonzero_morph = np.count_nonzero(morph)
print(f"morphology: 非零像素={nonzero_morph} (去除了{nonzero_count - nonzero_morph}个噪声像素)")

morph_labeled = put_label(morph, "形态学处理")
cv2.imwrite(os.path.join(save_dir, "morphology_frame.png"), morph_labeled)
print(f"已保存: morphology_frame.png")

# ========== 3. 修复轮廓提取（最近邻约束） ==========
print(f"\n--- 3. 修复轮廓提取 ---")

contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_contour = None
best_dist = float('inf')
for c in contours:
    M = cv2.moments(c)
    if M["m00"] == 0:
        continue
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
    if dist < best_dist:
        best_dist = dist
        best_contour = c

valid_contours = [best_contour] if best_contour is not None else []
print(f"总轮廓数: {len(contours)}, 有效轮廓数(最近邻): {len(valid_contours)}")

vis = test_frame.copy()
cv2.drawContours(vis, valid_contours, -1, (255, 0, 0), 2)

vis_labeled = put_label(vis, "轮廓提取")
cv2.imwrite(os.path.join(save_dir, "contour_frame.png"), vis_labeled)
print(f"已保存: contour_frame.png")

# ========== 4. 修复轨迹图 ==========
print(f"\n--- 4. 修复轨迹图 ---")

pred_bboxes = []
traj_path = os.path.join(save_dir, "trajectory.csv")
with open(traj_path, 'rb') as f:
    raw = f.read()
try:
    text = raw.decode('utf-8-sig')
except:
    try:
        text = raw.decode('gbk')
    except:
        text = raw.decode('utf-8', errors='replace')

pred_trajectory = []
lines = text.strip().split('\n')
for line in lines[1:]:
    parts = line.strip().split(',')
    if len(parts) >= 5:
        bx, by, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if bw > 10 and bh > 10:
            pred_center = (int(bx + bw / 2), int(by + bh / 2))
            pred_trajectory.append(pred_center)
        elif pred_trajectory:
            pred_trajectory.append(pred_trajectory[-1])
        else:
            pred_trajectory.append((int(prev_cx), int(prev_cy)))

gt_centers = []
for b in gt:
    gt_centers.append([b[0] + b[2] / 2, b[1] + b[3] / 2])
gt_centers = np.array(gt_centers)

print(f"预测轨迹点数: {len(pred_trajectory)}, GT帧数: {len(gt_centers)}")

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(gt_centers[:, 0], gt_centers[:, 1], 'g--', label='真实轨迹', linewidth=2)

if len(pred_trajectory) > 1:
    xs, ys = zip(*pred_trajectory)
    ax.plot(xs, ys, 'r-', linewidth=4, label='预测轨迹')
    # 加粗红色预测线并加散点标记
    ax.scatter(xs, ys, color='red', s=15, zorder=5)
    print(f"  已绘制红色预测轨迹, {len(xs)}个点")
    unique_pts = set(zip(xs, ys))
    if len(unique_pts) == 1:
        print(f"  注意: 所有预测点重合于 {(list(unique_pts)[0])}")

if len(pred_trajectory) > 0:
    ax.scatter(pred_trajectory[0][0], pred_trajectory[0][1], color='blue', s=200,
               marker='o', zorder=10, label='起始点', edgecolors='black', linewidths=2)

ax.set_xlabel('X坐标', fontproperties=font)
ax.set_ylabel('Y坐标', fontproperties=font)
ax.set_title(f'颜色阈值法 - 目标轨迹', fontproperties=font)
ax.legend(prop=font)
ax.grid(True)
ax.invert_yaxis()

plt.tight_layout()
traj_output = os.path.join(save_dir, f'{seq_name}_{method_name}_trajectory.png')
plt.savefig(traj_output, dpi=100)
plt.close()
print(f"已保存: trajectory.png")

print(f"\n完成! 所有4张图已生成到: {save_dir}")