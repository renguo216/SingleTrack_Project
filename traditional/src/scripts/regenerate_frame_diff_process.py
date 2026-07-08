"""
重新生成帧差法的中间处理过程合图（去掉红框，改进视觉效果）。
保存为另一张图片，不覆盖原文件。
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

seq_name = "GOT-10k_Val_000143"
input_dir = f"D:\\experiment\\SingleTrack_Project\\data\\datasets\\GOT_test\\{seq_name}"
output_dir = f"D:\\experiment\\SingleTrack_Project\\results\\traditional\\{seq_name}\\frame_diff"
os.makedirs(output_dir, exist_ok=True)

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

frames, gt = load_got_sequence(input_dir)
if len(frames) < 2:
    print("Not enough frames")
    exit(1)

frame1 = frames[0]
frame2 = frames[1]
bbox = gt[0]
x, y, w, h = [int(v) for v in bbox]

# ========== 1. 帧差图（彩色显示） ==========
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

diff = cv2.absdiff(gray1, gray2)
# 用热力图彩色映射让帧差更明显
diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

# ========== 2. 二值化图 ==========
_, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# ========== 3. 轮廓图（去掉红框，绿色轮廓加粗） ==========
kernel = np.ones((3, 3), np.uint8)
diff_morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_CLOSE, kernel)

contour_img = frame2.copy()
contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)  # 加粗到3

# ========== 合并 ==========
# 统一尺寸
max_height = max(diff_colored.shape[0], binary_colored.shape[0], contour_img.shape[0])
max_width = max(diff_colored.shape[1], binary_colored.shape[1], contour_img.shape[1])

font_size = int(max_height * 0.08)
text_height = font_size + 20

resized_images = []
for img in [diff_colored, binary_colored, contour_img]:
    resized = cv2.resize(img, (max_width, max_height))
    resized_images.append(resized)

total_width = max_width * 3
total_height = max_height + text_height

merged = np.zeros((total_height, total_width, 3), dtype=np.uint8)
merged.fill(255)

for i, img in enumerate(resized_images):
    merged[:max_height, i*max_width:(i+1)*max_width] = img

merged_pil = Image.fromarray(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(merged_pil)

try:
    font = ImageFont.truetype("simhei.ttf", font_size)
except:
    try:
        font = ImageFont.truetype("simsun.ttc", font_size)
    except:
        font = ImageFont.load_default()

labels = ["帧差", "二值化", "轮廓"]
for i, label in enumerate(labels):
    x_center = i * max_width + max_width // 2
    y = max_height + 10
    text_bbox = draw.textbbox((x_center, y), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = x_center - text_width // 2
    draw.text((text_x, y), label, fill=(0, 0, 0), font=font)

merged_cv = cv2.cvtColor(np.array(merged_pil), cv2.COLOR_RGB2BGR)
output_path = os.path.join(output_dir, "frame_diff_process_v2.png")
cv2.imwrite(output_path, merged_cv)
print(f"Saved: {output_path}")