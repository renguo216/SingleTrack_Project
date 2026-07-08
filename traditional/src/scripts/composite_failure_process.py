"""
将两个失败案例的处理图分别横向拼接，每张子图下方标注对应的处理步骤名称。
案例1: frame_diff.png + binary.png → "帧差" + "二值化"
案例3: foreground_mask.png + morphology.png → "前景掩码" + "形态学处理"
字号翻倍。
"""
import os
from PIL import Image, ImageDraw, ImageFont

font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"]
font = None
for fp in font_paths:
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 72)
        break
if font is None:
    font = ImageFont.load_default()

# ========== 案例1：帧差法 ==========
base1 = r'd:/experiment/SingleTrack_Project/results/failure_analysis/GOT-10k_Val_000146_frame_diff'
output1 = os.path.join(base1, 'process.png')

files1 = [('frame_diff.png', '帧差'), ('binary.png', '二值化')]
images1 = []
for fn, _ in files1:
    images1.append(Image.open(os.path.join(base1, fn)).convert("RGB"))

w, h = images1[0].size
label_h = 110
total_w = w * 2
total_h = h + label_h

composite1 = Image.new("RGB", (total_w, total_h), (255, 255, 255))
draw1 = ImageDraw.Draw(composite1)

for i, (img, label) in enumerate(zip(images1, [lb for _, lb in files1])):
    composite1.paste(img, (i * w, 0))
    bbox = draw1.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = i * w + (w - tw) // 2
    ty = h + (label_h - (bbox[3] - bbox[1])) // 2
    draw1.text((tx, ty), label, fill=(0, 0, 0), font=font)

composite1.save(output1, quality=95)
print(f"已保存: {output1}")

# ========== 案例3：背景减除法 ==========
base2 = r'd:/experiment/SingleTrack_Project/results/failure_analysis/GOT-10k_Val_000153_bg_subtract'
output2 = os.path.join(base2, 'process.png')

files2 = [('foreground_mask.png', '前景掩码'), ('morphology.png', '形态学处理')]
images2 = []
for fn, _ in files2:
    images2.append(Image.open(os.path.join(base2, fn)).convert("RGB"))

w2, h2 = images2[0].size
total_w2 = w2 * 2
total_h2 = h2 + label_h

composite2 = Image.new("RGB", (total_w2, total_h2), (255, 255, 255))
draw2 = ImageDraw.Draw(composite2)

for i, (img, label) in enumerate(zip(images2, [lb for _, lb in files2])):
    composite2.paste(img, (i * w2, 0))
    bbox = draw2.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = i * w2 + (w2 - tw) // 2
    ty = h2 + (label_h - (bbox[3] - bbox[1])) // 2
    draw2.text((tx, ty), label, fill=(0, 0, 0), font=font)

composite2.save(output2, quality=95)
print(f"已保存: {output2}")

print("\n全部完成!")