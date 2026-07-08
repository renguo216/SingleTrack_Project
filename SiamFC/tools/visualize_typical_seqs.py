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
from matplotlib.patches import Rectangle


DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_DIR = r'result\typical_seqs'
SELECTED_SEQS = ['Basketball', 'David']


def get_img_files(seq_dir):
    img_dir = os.path.join(seq_dir, 'img')
    if os.path.isdir(img_dir):
        files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if len(files) > 0:
            return files
    files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    return files


def visualize_sequence(seq_name):
    """生成单个序列的典型帧展示图"""
    seq_dir = os.path.join(DATASET_ROOT, seq_name)
    img_files = get_img_files(seq_dir)

    anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
    try:
        try:
            anno = np.loadtxt(anno_file, delimiter=',')
        except:
            anno = np.loadtxt(anno_file)
    except:
        print('无法加载 %s 的标注' % seq_name)
        return

    n = min(len(img_files), len(anno))

    # ===== 图1: 首帧目标框 + 局部放大 =====
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('典型序列: %s - 首帧目标框与局部放大' % seq_name, fontsize=14)

    # 首帧全图
    img0 = cv2.imread(img_files[0])
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    ax = axes1[0]
    box0 = anno[0]
    ax.add_patch(Rectangle((box0[0], box0[1]), box0[2], box0[3],
                           linewidth=2, edgecolor='red', facecolor='none'))
    ax.imshow(img0_rgb)
    ax.set_title('首帧: 目标框 [%.0f, %.0f, %.0f, %.0f]' %
                 (box0[0], box0[1], box0[2], box0[3]))
    ax.axis('off')

    # 局部放大
    ax = axes1[1]
    cx = box0[0] + box0[2] / 2
    cy = box0[1] + box0[3] / 2
    zoom_sz = max(box0[2], box0[3]) * 3
    x1 = max(0, int(cx - zoom_sz / 2))
    y1 = max(0, int(cy - zoom_sz / 2))
    x2 = min(img0_rgb.shape[1], int(cx + zoom_sz / 2))
    y2 = min(img0_rgb.shape[0], int(cy + zoom_sz / 2))
    zoom_img = img0_rgb[y1:y2, x1:x2].copy()
    box_local = [box0[0] - x1, box0[1] - y1, box0[2], box0[3]]
    ax.add_patch(Rectangle((box_local[0], box_local[1]), box_local[2], box_local[3],
                           linewidth=2, edgecolor='red', facecolor='none'))
    ax.imshow(zoom_img)
    ax.set_title('局部放大 (目标区域)')
    ax.axis('off')

    plt.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, '%s_frame1.png' % seq_name),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print('已保存: %s_frame1.png' % seq_name)

    # ===== 图2: 6个关键帧（上三下三） =====
    key_indices = np.linspace(0, n - 1, 3, dtype=int)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('典型序列: %s - 关键帧序列 (共%d帧)' % (seq_name, n), fontsize=14)

    for i, fidx in enumerate(key_indices):
        ax = axes2[i]
        img = cv2.imread(img_files[fidx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        box = anno[fidx]
        ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3],
                               linewidth=2, edgecolor='red', facecolor='none'))
        ax.imshow(img_rgb)
        ax.set_title('帧%d' % (fidx + 1))
        ax.axis('off')

    plt.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, '%s_keyframes.png' % seq_name),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('已保存: %s_keyframes.png' % seq_name)

    # ===== 文本说明 =====
    start_cx = anno[0][0] + anno[0][2] / 2
    start_cy = anno[0][1] + anno[0][3] / 2
    end_cx = anno[-1][0] + anno[-1][2] / 2
    end_cy = anno[-1][1] + anno[-1][3] / 2
    dist = np.sqrt((end_cx - start_cx) ** 2 + (end_cy - start_cy) ** 2)

    info = (
        '典型序列: %s\n'
        '总帧数: %d\n\n'
        '首帧目标框: [x=%.1f, y=%.1f, w=%.1f, h=%.1f]\n'
        '末帧目标框: [x=%.1f, y=%.1f, w=%.1f, h=%.1f]\n'
        '目标移动距离: %.1f 像素\n\n'
        '关键帧索引:\n' % (seq_name, n,
                        anno[0][0], anno[0][1], anno[0][2], anno[0][3],
                        anno[-1][0], anno[-1][1], anno[-1][2], anno[-1][3], dist))
    for i, fidx in enumerate(key_indices):
        info += '  帧%d: 目标框 [%.1f, %.1f, %.1f, %.1f]\n' % (
            fidx + 1, anno[fidx][0], anno[fidx][1], anno[fidx][2], anno[fidx][3])

    txt_path = os.path.join(OUTPUT_DIR, '%s_info.txt' % seq_name)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(info)
    print('已保存: %s_info.txt' % seq_name)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print('生成典型序列可视化...')
    for seq_name in SELECTED_SEQS:
        visualize_sequence(seq_name)
    print('\n完成! 图片保存在 %s/' % OUTPUT_DIR)


if __name__ == '__main__':
    main()