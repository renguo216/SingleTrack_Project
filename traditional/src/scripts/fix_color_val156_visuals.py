"""
修复颜色阈值法在GOT-10k_Val_000156（蜗牛）序列上的中间结果图。
生成 color_mask.png、morphology_frame.png、contour_frame.png。
"""
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

seq_name = "GOT-10k_Val_000156"
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


# ========== 加载序列 ==========
def load_got_sequence(seq_path):
    frames = []
    for f in sorted(os.listdir(seq_path)):
        if f.endswith(('.jpg', '.png')):
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

# ========== 取展示帧（第1帧） ==========
test_idx = 0
test_frame = frames[test_idx].copy()
bbox = gt[test_idx]
x, y, w, h = [int(v) for v in bbox]
initial_area = w * h
prev_cx, prev_cy = x + w / 2, y + h / 2
print(f"首帧GT框: ({x}, {y}, {w}, {h}), 面积={initial_area}")

# ========== 1. 修复颜色掩码（中心区域采样+收紧HSV） ==========
print(f"\n--- 1. 修复颜色掩码 ---")

margin_x, margin_y = int(w * 0.2), int(h * 0.2)
roi = test_frame[y + margin_y:y + h - margin_y, x + margin_x:x + w - margin_x]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 收紧HSV：S下限≥40，V上限≤200
h_min = max(0, int(np.percentile(hsv_roi[:, :, 0], 5)) - 5)
h_max = min(180, int(np.percentile(hsv_roi[:, :, 0], 95)) + 5)
s_min = max(40, int(np.percentile(hsv_roi[:, :, 1], 25)))
s_max = min(255, int(np.percentile(hsv_roi[:, :, 1], 90)) + 10)
v_min = max(20, int(np.percentile(hsv_roi[:, :, 2], 10)) - 10)
v_max = min(200, int(np.percentile(hsv_roi[:, :, 2], 90)) + 10)

lower = (h_min, s_min, v_min)
upper = (h_max, s_max, v_max)
print(f"中心区域采样: roi={roi.shape[1]}x{roi.shape[0]}")
print(f"HSV范围: lower={lower}, upper={upper}")

hsv_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2HSV)
color_mask = cv2.inRange(hsv_frame, lower, upper)
color_mask = color_mask.astype(np.uint8)

# 轻微形态学清理（去孤立点）
tiny_kernel = np.ones((3, 3), np.uint8)
color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, tiny_kernel, iterations=1)

nonzero_count = np.count_nonzero(color_mask)
total_pixels = color_mask.shape[0] * color_mask.shape[1]
print(f"color_mask: 非零像素={nonzero_count} ({nonzero_count/total_pixels*100:.1f}%)")

color_mask_labeled = put_label(color_mask, "颜色掩码")
cv2.imwrite(os.path.join(save_dir, "color_mask.png"), color_mask_labeled)
print(f"已保存: color_mask.png")

# ========== 2. 修复形态学处理（大核+连通域过滤） ==========
print(f"\n--- 2. 修复形态学处理 ---")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
morph = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=3)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=3)

# 连通域面积过滤
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

# 只保留距离初始目标框中心最近的一个轮廓
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

valid_contours = []
if best_contour is not None:
    valid_contours = [best_contour]

print(f"总轮廓数: {len(contours)}, 有效轮廓数(最近邻): {len(valid_contours)}")

vis = test_frame.copy()
cv2.drawContours(vis, valid_contours, -1, (255, 0, 0), 2)

vis_labeled = put_label(vis, "轮廓提取")
cv2.imwrite(os.path.join(save_dir, "contour_frame.png"), vis_labeled)
print(f"已保存: contour_frame.png")

# ========== 验证 ==========
print(f"\n--- 验证 ---")
cm = cv2.imread(os.path.join(save_dir, 'color_mask.png'), cv2.IMREAD_GRAYSCALE)
print(f"color_mask: {cm.shape}, nonzero={np.count_nonzero(cm)}, ratio={np.count_nonzero(cm)/(cm.shape[0]*cm.shape[1])*100:.1f}%")
mp = cv2.imread(os.path.join(save_dir, 'morphology_frame.png'), cv2.IMREAD_GRAYSCALE)
print(f"morphology: {mp.shape}, nonzero={np.count_nonzero(mp)}")
ct = cv2.imread(os.path.join(save_dir, 'contour_frame.png'))
r2,g2,b2 = ct[:,:,2], ct[:,:,1], ct[:,:,0]
blue = np.sum((b2>150)&(r2<80)&(g2<80))
print(f"contour: blue pixels={blue}")

print(f"\n完成! 所有文件保存到 {save_dir}/")