"""
将 GOT-10k_Train_000424/edge_contour 下的 5 张 frame_ 图片横向排列成一行，
检测图中已有的黄色GT框和红色Edge框并加粗，
标注对应帧号，缩放到 9600×1200。
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 路径设置
input_dir = r"d:/experiment/SingleTrack_Project/results/traditional/GOT-10k_Train_000424/edge_contour"
output_path = os.path.join(input_dir, "composite.png")

# 5 张图的文件名（按帧顺序）
file_names = [
    "frame_0000.png",
    "frame_0025.png",
    "frame_0050.png",
    "frame_0075.png",
    "frame_0099.png",
]

# 对应的标签
labels = ["第1帧", "第15帧", "第30帧", "第45帧", "第60帧"]

# 线宽（加粗）
line_width = 20
expand = 8

# 加载并处理图片
processed = []
for fn in file_names:
    fp = os.path.join(input_dir, fn)
    img = Image.open(fp).convert("RGB")
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    draw = ImageDraw.Draw(img)
    
    # 检测黄色像素 (GT框颜色: 0,255,255)
    yellow_mask = (g > 200) & (r > 200) & (b < 80)
    # 检测红色像素 (Edge框颜色: 0,0,255)
    red_mask = (r > 180) & (g < 80) & (b < 80)
    
    # 加粗黄色GT框
    if np.any(yellow_mask):
        ys, xs = np.where(yellow_mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        for t in range(line_width):
            draw.line([(x_min - expand, y_min - expand + t), (x_max + expand, y_min - expand + t)], fill=(0, 255, 255))
            draw.line([(x_min - expand, y_max + expand - t), (x_max + expand, y_max + expand - t)], fill=(0, 255, 255))
            draw.line([(x_min - expand + t, y_min - expand), (x_min - expand + t, y_max + expand)], fill=(0, 255, 255))
            draw.line([(x_max + expand - t, y_min - expand), (x_max + expand - t, y_max + expand)], fill=(0, 255, 255))
    
    # 加粗红色Edge框
    if np.any(red_mask):
        ys, xs = np.where(red_mask)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        for t in range(line_width):
            draw.line([(x_min - expand, y_min - expand + t), (x_max + expand, y_min - expand + t)], fill=(255, 0, 0))
            draw.line([(x_min - expand, y_max + expand - t), (x_max + expand, y_max + expand - t)], fill=(255, 0, 0))
            draw.line([(x_min - expand + t, y_min - expand), (x_min - expand + t, y_max + expand)], fill=(255, 0, 0))
            draw.line([(x_max + expand - t, y_min - expand), (x_max + expand - t, y_max + expand)], fill=(255, 0, 0))
    
    processed.append(img)

# 取第一张的尺寸
w, h = processed[0].size

# 创建横向拼接的大图（下方留空间给标签）
label_height = 80
total_width = w * 5
total_height = h + label_height

composite = Image.new("RGB", (total_width, total_height), (255, 255, 255))
draw = ImageDraw.Draw(composite)

# 加载中文字体（大一点）
font_paths = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/yahei.ttf",
]
font = None
for fp in font_paths:
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 56)
        break
if font is None:
    font = ImageFont.load_default()

# 拼接图片并添加标签（图片之间无缝隙）
for i, pi in enumerate(processed):
    x_offset = i * w
    composite.paste(pi, (x_offset, 0))
    label = labels[i]
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = x_offset + (w - tw) // 2
    ty = h + (label_height - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

# 缩放到指定尺寸
target_size = (9600, 1200)
composite = composite.resize(target_size, Image.LANCZOS)
print(f"已缩放至: {composite.size}")

composite.save(output_path, quality=95)
print(f"合成图片已保存到: {output_path}")
print(f"图片尺寸: {composite.size}")