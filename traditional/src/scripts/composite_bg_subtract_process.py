"""
将 background_model.png、foreground_mask.png、morphology_frame.png
三张图片横向拼接成一行，每张图下方居中标注方法名称。
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 路径设置
base_dir = r"d:/experiment/SingleTrack_Project/results/traditional/GOT-10k_Val_000166/bg_subtract"
output_path = os.path.join(base_dir, "bg_subtract_process.png")

file_names = [
    "background_model.png",
    "foreground_mask.png",
    "morphology_frame.png",
]

# 对应中文标签
labels = ["背景模型", "前景掩码", "形态学处理"]

# 加载图片
images = []
for fn in file_names:
    fp = os.path.join(base_dir, fn)
    img = Image.open(fp).convert("RGB")
    images.append(img)

# 取第一张的尺寸作为标准
w, h = images[0].size

# 下方标签高度（字号翻倍后相应增加）
label_height = 100

# 总尺寸
total_width = w * 3
total_height = h + label_height

# 创建白色画布
composite = Image.new("RGB", (total_width, total_height), (255, 255, 255))
draw = ImageDraw.Draw(composite)

# 加载中文字体（字号翻倍：36→72）
font_paths = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]
font = None
for fp in font_paths:
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 72)
        break
if font is None:
    font = ImageFont.load_default()

# 拼接图片并添加标签
for i, img in enumerate(images):
    x_offset = i * w
    # 粘贴图片
    composite.paste(img, (x_offset, 0))
    # 绘制标签（居中显示在底部区域）
    label = labels[i]
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = x_offset + (w - tw) // 2
    ty = h + (label_height - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

# 缩放到 3840×1200（保持内容比例，上下加白边填充）
target_size = (3840, 1200)
canvas = Image.new("RGB", target_size, (255, 255, 255))
# 按比例缩放原图，使其宽度/高度不超过目标尺寸
composite.thumbnail(target_size, Image.LANCZOS)
# 居中粘贴
pw, ph = composite.size
px = (target_size[0] - pw) // 2
py = (target_size[1] - ph) // 2
canvas.paste(composite, (px, py))

canvas.save(output_path, quality=95)
print(f"合成图片已保存到: {output_path}")
print(f"图片尺寸: {canvas.size}")
