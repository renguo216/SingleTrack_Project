from __future__ import absolute_import, print_function, division

import os
import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from siamfc import ops


DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_DIR = r'result\preprocess'
SELECTED_SEQS = ['Basketball', 'David']

EXEMPLAR_SZ = 127
INSTANCE_SZ = 255
CONTEXT = 0.5


def get_img_files(seq_dir):
    img_dir = os.path.join(seq_dir, 'img')
    if os.path.isdir(img_dir):
        files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if len(files) > 0:
            return files
    files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    return files


def crop_for_vis(img, center, target_sz, out_size):
    context_sz = CONTEXT * np.sum(target_sz)
    z_sz = np.sqrt(np.prod(target_sz + context_sz))
    size = z_sz * out_size / EXEMPLAR_SZ
    avg_color = np.mean(img, axis=(0, 1))
    patch = ops.crop_and_resize(
        img, center, size, out_size,
        border_value=avg_color)
    return patch


def draw_box(img, box, color, thickness=2):
    x, y, w, h = map(int, box)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def process_sequence(seq_name):
    """处理单个序列，生成预处理流程演示图和文字说明"""
    seq_dir = os.path.join(DATASET_ROOT, seq_name)
    img_files = get_img_files(seq_dir)

    anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
    try:
        try:
            anno = np.loadtxt(anno_file, delimiter=',')
        except:
            anno = np.loadtxt(anno_file)
    except:
        print('无法加载 %s 的标注文件' % seq_name)
        return

    # 取首帧
    idx = 0
    img_bgr = cv2.imread(img_files[idx])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    box = anno[idx]
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    center = np.array([cx, cy])
    target_sz = box[2:].astype(np.float32)

    # ===== 图: 原图+目标框 | 模板帧 | 搜索区域 =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('SiamFC 预处理流程 - %s' % seq_name, fontsize=14)

    # 1. 原图+目标框
    ax = axes[0]
    img_vis = img_rgb.copy()
    draw_box(img_vis, box, (255, 0, 0), 3)
    ax.imshow(img_vis)
    ax.set_title('原始图像 + 目标框')
    ax.axis('off')

    # 2. 模板帧
    exemplar = crop_for_vis(img_rgb, center, target_sz, EXEMPLAR_SZ)
    ax = axes[1]
    ax.imshow(exemplar.astype(np.uint8))
    ax.set_title('模板帧 (%dx%d)' % (EXEMPLAR_SZ, EXEMPLAR_SZ))
    ax.axis('off')

    # 3. 搜索区域
    search = crop_for_vis(img_rgb, center, target_sz, INSTANCE_SZ)
    ax = axes[2]
    ax.imshow(search.astype(np.uint8))
    ax.set_title('搜索区域 (%dx%d)' % (INSTANCE_SZ, INSTANCE_SZ))
    ax.axis('off')

    plt.tight_layout()
    seq_output = os.path.join(OUTPUT_DIR, seq_name)
    if not os.path.exists(seq_output):
        os.makedirs(seq_output)

    img_path = os.path.join(seq_output, 'preprocess.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('已保存: %s' % img_path)

    # ===== 坐标转换说明文本 =====
    info = (
        'SiamFC 预处理流程 - %s\n'
        '========================\n\n'
        '帧索引: %d (首帧)\n\n'
        '1. 原始标注框 (角点格式 [x, y, w, h]):\n'
        '   [x=%.1f, y=%.1f, w=%.1f, h=%.1f]\n\n'
        '2. 转换为中心格式 [cx, cy, h, w]:\n'
        '   cx = x + w/2 = %.1f\n'
        '   cy = y + h/2 = %.1f\n'
        '   [cx=%.1f, cy=%.1f, h=%.1f, w=%.1f]\n\n'
        '3. 模板帧大小: %d x %d\n'
        '4. 搜索帧大小: %d x %d\n'
        '5. 上下文边距: %.1f (相对于目标尺寸)\n'
        '6. 颜色空间: RGB\n'
        '7. 像素范围: [0, 255]\n' % (
            seq_name, idx + 1,
            box[0], box[1], box[2], box[3],
            cx, cy,
            cx, cy, box[3], box[2],
            EXEMPLAR_SZ, EXEMPLAR_SZ,
            INSTANCE_SZ, INSTANCE_SZ,
            CONTEXT))

    txt_path = os.path.join(seq_output, 'preprocess_info.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(info)
    print('已保存: %s' % txt_path)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print('生成预处理流程可视化...')
    for seq_name in SELECTED_SEQS:
        process_sequence(seq_name)
    print('\n完成! 图片保存在 %s/' % OUTPUT_DIR)


if __name__ == '__main__':
    main()