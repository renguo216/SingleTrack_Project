"""
将 color_mask.png、morphology_frame.png、contour_frame.png
三张图片横向拼接成一行，每张图下方居中标注方法名称。
对 contour_frame.png 中的蓝色轮廓线条加粗。
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 路径设置
base_dir = r"d:/experiment/SingleTrack_Project/results/traditional/GOT-10k_Val_000146/color"
output_path = os.path.join(base_dir, "color_process.png")

file_names = [
    "color_mask.png",
    "morphology_frame.png",
    "contour_frame.png",
]

# 对应中文标签
labels = ["颜色掩码", "形态学处理", "轮廓提取"]

# 加载图片
images = []
for fn in file_names:
    fp = os.path.join(base_dir, fn)
    img = cv2.imread(fp, cv2.IMREAD_COLOR)
    
    # 对第三张图（contour_frame.png）加粗蓝色轮廓
    if fn == "contour_frame.png":
        r, g, b = img[:,:,2], img[:,:,1], img[:,:,0]
        # 检测蓝色像素
        blue_mask = (b > 150) & (r < 80) & (g < 80)
        # 对蓝色区域做膨胀加粗
        kernel = np.ones((7, 7), np.uint8)
        blue_mask_thick = cv2.dilate(blue_mask.astype(np.uint8), kernel, iterations=1)
        # 覆盖为更粗的蓝色线条
        img[blue_mask_thick > 0] = (0, 0, 255)  # BGR: 蓝色
    
    # 转为 PIL Image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    images.append(pil_img)

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
    composite.paste(img, (x_offset, 0))
    label = labels[i]
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = x_offset + (w - tw) // 2
    ty = h + (label_height - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)

# 缩放到 3840×1200
target_size = (3840, 1200)
canvas = Image.new("RGB", target_size, (255, 255, 255))
composite.thumbnail(target_size, Image.LANCZOS)
pw, ph = composite.size
px = (target_size[0] - pw) // 2
py = (target_size[1] - ph) // 2
canvas.paste(composite, (px, py))

canvas.save(output_path, quality=95)
print(f"合成图片已保存到: {output_path}")
print(f"图片尺寸: {canvas.size}")