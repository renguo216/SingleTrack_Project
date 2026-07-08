"""
为3个失败案例分别生成失败帧的可视化（GT框 vs 预测框对比）和中间处理过程图。
保存到 results/failure_analysis/
"""
import os
import sys
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

out_dir = r'd:/experiment/SingleTrack_Project/results/failure_analysis'

def put_label(img_cv2, text, pos=(10, 30), color=(0, 255, 0), size=28):
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]
    fnt = None
    for fp in font_paths:
        if os.path.exists(fp):
            fnt = ImageFont.truetype(fp, size)
            break
    if fnt is None:
        fnt = ImageFont.load_default()
    draw.text(pos, text, fill=color, font=fnt)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ========== 案例1：帧差法在 Val_000146（目标静止致帧差失效）==========
print("=== 案例1: 帧差法在 GOT-10k_Val_000146 ===")
seq = "GOT-10k_Val_000146"
method = "frame_diff"
input_dir = f"D:/experiment/SingleTrack_Project/data/datasets/GOT_test/{seq}"
res_dir = f"D:/experiment/SingleTrack_Project/results/traditional/{seq}/{method}"
save_dir = os.path.join(out_dir, f"{seq}_{method}")
os.makedirs(save_dir, exist_ok=True)

frames = []
for f in sorted(os.listdir(input_dir)):
    if f.endswith(('.jpg','.png')):
        img = cv2.imread(os.path.join(input_dir, f))
        if img is not None:
            frames.append(img)

gt = []
with open(os.path.join(input_dir, 'groundtruth.txt'), 'r') as f:
    for line in f:
        p = line.strip().split(',')
        gt.append((int(float(p[0])), int(float(p[1])), int(float(p[2])), int(float(p[3]))))

pred = []
with open(os.path.join(res_dir, 'trajectory.csv'), 'rb') as f:
    text = f.read().decode('utf-8-sig')
for line in text.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    pred.append((int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))))

frame_idx = 30
frame = frames[frame_idx].copy()
vis = frame.copy()
gx, gy, gw, gh = gt[frame_idx]
cv2.rectangle(vis, (gx, gy), (gx+gw, gy+gh), (0, 255, 255), 4)
cv2.putText(vis, 'GT', (gx, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
px, py, pw, ph = pred[frame_idx]
cv2.rectangle(vis, (px, py), (px+pw, py+ph), (0, 0, 255), 4)
cv2.putText(vis, 'Pred', (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
vis = put_label(vis, f"帧差法失败案例 - 第{frame_idx+1}帧", (10, 60), (0, 255, 255), 32)
cv2.imwrite(os.path.join(save_dir, 'frame_对比.png'), vis)

# 帧差中间过程
gray1 = cv2.cvtColor(frames[frame_idx-1], cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5,5), 0)
gray2 = cv2.GaussianBlur(gray2, (5,5), 0)
diff = cv2.absdiff(gray1, gray2)
diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
diff_bgr = put_label(diff_bgr, "帧差结果（目标不动，几乎全黑）", (10, 30), (0, 255, 0), 24)
cv2.imwrite(os.path.join(save_dir, 'frame_diff.png'), diff_bgr)

_, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
binary_bgr = put_label(binary_bgr, "二值化（几乎无前景）", (10, 30), (0, 255, 0), 24)
cv2.imwrite(os.path.join(save_dir, 'binary.png'), binary_bgr)
print("已保存: frame_对比.png, frame_diff.png, binary.png")


# ========== 案例2：颜色阈值法在 Val_000151（大目标颜色混淆）==========
print("\n=== 案例2: 颜色阈值法在 GOT-10k_Val_000151 ===")
seq = "GOT-10k_Val_000151"
method = "color"
input_dir = f"D:/experiment/SingleTrack_Project/data/datasets/GOT_test/{seq}"
res_dir = f"D:/experiment/SingleTrack_Project/results/traditional/{seq}/{method}"
save_dir = os.path.join(out_dir, f"{seq}_{method}")
os.makedirs(save_dir, exist_ok=True)

frames = []
for f in sorted(os.listdir(input_dir)):
    if f.endswith(('.jpg','.png')):
        img = cv2.imread(os.path.join(input_dir, f))
        if img is not None:
            frames.append(img)

gt = []
with open(os.path.join(input_dir, 'groundtruth.txt'), 'r') as f:
    for line in f:
        p = line.strip().split(',')
        gt.append((int(float(p[0])), int(float(p[1])), int(float(p[2])), int(float(p[3]))))

pred = []
with open(os.path.join(res_dir, 'trajectory.csv'), 'rb') as f:
    text = f.read().decode('utf-8-sig')
for line in text.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    pred.append((int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))))

frame_idx = 30
frame = frames[frame_idx].copy()
vis = frame.copy()
gx, gy, gw, gh = gt[frame_idx]
cv2.rectangle(vis, (gx, gy), (gx+gw, gy+gh), (0, 255, 255), 4)
cv2.putText(vis, 'GT', (gx, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
px, py, pw, ph = pred[frame_idx]
cv2.rectangle(vis, (px, py), (px+pw, py+ph), (0, 0, 255), 4)
cv2.putText(vis, 'Pred', (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
vis = put_label(vis, f"颜色阈值法失败案例 - 第{frame_idx+1}帧", (10, 60), (0, 255, 255), 32)
cv2.imwrite(os.path.join(save_dir, 'frame_对比.png'), vis)

# 颜色概率图
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
roi = hsv[gy:gy+gh, gx:gx+gw]
mask = cv2.inRange(roi, (0, 30, 30), (180, 255, 255))
hist = cv2.calcHist([roi], [0, 1], mask, [45, 64], [0, 180, 0, 256])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
prob = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
prob_bgr = cv2.cvtColor(prob, cv2.COLOR_GRAY2BGR)
prob_bgr = put_label(prob_bgr, "颜色概率图（目标与背景混淆）", (10, 30), (0, 255, 0), 24)
cv2.imwrite(os.path.join(save_dir, 'color_prob.png'), prob_bgr)
print("已保存: frame_对比.png, color_prob.png")


# ========== 案例3：背景减除法在 Val_000153（背景运动干扰）==========
print("\n=== 案例3: 背景减除法在 GOT-10k_Val_000153 ===")
seq = "GOT-10k_Val_000153"
method = "bg_subtract"
input_dir = f"D:/experiment/SingleTrack_Project/data/datasets/GOT_test/{seq}"
res_dir = f"D:/experiment/SingleTrack_Project/results/traditional/{seq}/{method}"
save_dir = os.path.join(out_dir, f"{seq}_{method}")
os.makedirs(save_dir, exist_ok=True)

frames = []
for f in sorted(os.listdir(input_dir)):
    if f.endswith(('.jpg','.png')):
        img = cv2.imread(os.path.join(input_dir, f))
        if img is not None:
            frames.append(img)

gt = []
with open(os.path.join(input_dir, 'groundtruth.txt'), 'r') as f:
    for line in f:
        p = line.strip().split(',')
        gt.append((int(float(p[0])), int(float(p[1])), int(float(p[2])), int(float(p[3]))))

pred = []
with open(os.path.join(res_dir, 'trajectory.csv'), 'rb') as f:
    text = f.read().decode('utf-8-sig')
for line in text.strip().split('\n')[1:]:
    parts = line.strip().split(',')
    pred.append((int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))))

frame_idx = 40
frame = frames[frame_idx].copy()
vis = frame.copy()
gx, gy, gw, gh = gt[frame_idx]
cv2.rectangle(vis, (gx, gy), (gx+gw, gy+gh), (0, 255, 255), 4)
cv2.putText(vis, 'GT', (gx, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
px, py, pw, ph = pred[frame_idx]
cv2.rectangle(vis, (px, py), (px+pw, py+ph), (0, 0, 255), 4)
cv2.putText(vis, 'Pred', (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
vis = put_label(vis, f"背景减除法失败案例 - 第{frame_idx+1}帧", (10, 60), (0, 255, 255), 32)
cv2.imwrite(os.path.join(save_dir, 'frame_对比.png'), vis)

# MOG2过程
bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=False)
for i in range(min(10, frame_idx)):
    bg.apply(frames[i])
fg = bg.apply(frame)
_, fg_bin = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
fg_bgr = cv2.cvtColor(fg_bin, cv2.COLOR_GRAY2BGR)
fg_bgr = put_label(fg_bgr, "MOG2前景掩码（大量误检）", (10, 30), (0, 255, 0), 24)
cv2.imwrite(os.path.join(save_dir, 'foreground_mask.png'), fg_bgr)

kernel = np.ones((5,5), np.uint8)
morph = cv2.morphologyEx(fg_bin, cv2.MORPH_OPEN, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)
morph_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
morph_bgr = put_label(morph_bgr, "形态学处理(噪声未完全去除)", (10, 30), (0, 255, 0), 24)
cv2.imwrite(os.path.join(save_dir, 'morphology.png'), morph_bgr)
print("已保存: frame_对比.png, foreground_mask.png, morphology.png")

print("\n全部完成！")