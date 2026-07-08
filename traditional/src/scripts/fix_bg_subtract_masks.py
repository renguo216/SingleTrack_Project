"""
修复背景减除法生成的 foreground_mask.png 和 morphology_frame.png 纯黑问题。
使用 MOG2 背景减除法（与 background_subtraction_tracker.py 一致）。
按用户提供的排查步骤进行调试和修复，并添加中文标签。
"""
import cv2
import numpy as np
import os

seq_name = "GOT-10k_Val_000166"
save_dir = f"D:\\experiment\\SingleTrack_Project\\results\\traditional\\{seq_name}\\bg_subtract"
input_dir = f"D:\\experiment\\SingleTrack_Project\\data\\datasets\\GOT_test\\{seq_name}"
os.makedirs(save_dir, exist_ok=True)


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


def put_chinese_label(img_cv2, text, pos=(10, 30)):
    """用 PIL 在 cv2 图像左上角加中文标签"""
    from PIL import Image, ImageDraw, ImageFont
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 28)
            break
    if font is None:
        font = ImageFont.load_default()

    draw.text(pos, text, fill=(0, 255, 0), font=font)  # 绿色标签
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


print(f"加载序列: {seq_name}")
frames, gt = load_got_sequence(input_dir)
print(f"共 {len(frames)} 帧, {len(gt)} 个标注框")

if len(frames) < 2:
    print("帧数不足")
    exit(1)

# ============ 使用MOG2背景减除法 ============
print(f"\n--- 初始化MOG2背景减除法 ---")

bg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=30,
    detectShadows=False
)

# 前10帧初始化背景模型
print("前10帧初始化背景模型...")
for i in range(min(10, len(frames))):
    bg.apply(frames[i])

# 取第30帧作为展示帧
test_idx = 29
if test_idx >= len(frames):
    test_idx = len(frames) - 1

test_frame = frames[test_idx].copy()
print(f"\n展示帧: 第{test_idx+1}帧")

# MOG2 生成前景掩码
fg = bg.apply(test_frame)

print(f"\n--- 步骤1: 调试打印 ---")
print(f"[DEBUG] fg mask: dtype={fg.dtype}, min={fg.min()}, max={fg.max()}, shape={fg.shape}")
print(f"[DEBUG] fg nonzero: {np.count_nonzero(fg)}")
print(f"[DEBUG] fg > 0: {np.count_nonzero(fg > 0)}")
print(f"[DEBUG] fg > 127: {np.count_nonzero(fg > 127)}")

# ============ 情况A: 检查最大值 ============
if fg.max() == 0:
    print(f"\n*** 情况A: fg.max()=0，MOG2参数需要调整 ***")
    print("降低 varThreshold 到 16...")
    bg2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
    for i in range(min(10, len(frames))):
        bg2.apply(frames[i])
    fg = bg2.apply(test_frame)
    print(f"  varThreshold=16: max={fg.max()}, nonzero={np.count_nonzero(fg>0)}")

    if fg.max() == 0:
        print("降低 varThreshold 到 10...")
        bg3 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=10, detectShadows=False)
        for i in range(min(10, len(frames))):
            bg3.apply(frames[i])
        fg = bg3.apply(test_frame)
        print(f"  varThreshold=10: max={fg.max()}, nonzero={np.count_nonzero(fg>0)}")

# ============ 情况B/C: 检查数据类型并修复二值化阈值 ============
print(f"\n--- 步骤2/3: 修复二值化阈值 ---")
print(f"fg.max() = {fg.max()}")

if fg.max() < 127:
    print(f"fg.max()={fg.max()} < 127，说明MOG2输出值范围很小")
    print(f"使用自适应阈值")
    adaptive_thresh = max(10, fg.max() // 2)
    _, fg_vis = cv2.threshold(fg, adaptive_thresh, 255, cv2.THRESH_BINARY)
    print(f"自适应阈值 = {adaptive_thresh}")
else:
    print("直接使用阈值 127")
    _, fg_vis = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)

# 确保 uint8 类型
fg_vis = fg_vis.astype(np.uint8)
print(f"[DEBUG] fg_vis: dtype={fg_vis.dtype}, min={fg_vis.min()}, max={fg_vis.max()}, nonzero={np.count_nonzero(fg_vis)}")

# 转 BGR 以便加标签
fg_vis_bgr = cv2.cvtColor(fg_vis, cv2.COLOR_GRAY2BGR)
fg_vis_bgr = put_chinese_label(fg_vis_bgr, "前景掩码")

fg_path = os.path.join(save_dir, "foreground_mask.png")
cv2.imwrite(fg_path, fg_vis_bgr)
print(f"已保存: {fg_path}")

# ============ 形态学处理 ============
print(f"\n--- 步骤4: 形态学处理 ---")
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(fg_vis, cv2.MORPH_OPEN, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
print(f"[DEBUG] morph: dtype={morph.dtype}, min={morph.min()}, max={morph.max()}, nonzero={np.count_nonzero(morph)}")

morph_bgr = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
morph_bgr = put_chinese_label(morph_bgr, "形态学处理")

morph_path = os.path.join(save_dir, "morphology_frame.png")
cv2.imwrite(morph_path, morph_bgr)
print(f"已保存: {morph_path}")

# ============ 验证 ============
print(f"\n--- 步骤5: 验证 ---")
fg_check = cv2.imread(fg_path, cv2.IMREAD_GRAYSCALE)
print(f"[验证] foreground_mask.png: min={fg_check.min()}, max={fg_check.max()}, nonzero={np.count_nonzero(fg_check)}")
morph_check = cv2.imread(morph_path, cv2.IMREAD_GRAYSCALE)
print(f"[验证] morphology_frame.png: min={morph_check.min()}, max={morph_check.max()}, nonzero={np.count_nonzero(morph_check)}")

print(f"\n完成!")