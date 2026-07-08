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
OUTPUT_DIR = r'result\augment'
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


def crop_patch(img, box, out_size):
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    center = np.array([cx, cy])
    target_sz = box[2:].astype(np.float32)
    context_sz = CONTEXT * np.sum(target_sz)
    z_sz = np.sqrt(np.prod(target_sz + context_sz))
    size = z_sz * out_size / EXEMPLAR_SZ
    avg_color = np.mean(img, axis=(0, 1))
    patch = ops.crop_and_resize(
        img, center, size, out_size,
        border_value=avg_color)
    return patch


def process_sequence(seq_name):
    """处理单个序列，生成数据增强效果图"""
    np.random.seed(42)

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

    idx = 0
    img_bgr = cv2.imread(img_files[idx])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    box = anno[idx]

    # 搜索区域进行增强演示
    search_patch = crop_patch(img_rgb, box, INSTANCE_SZ)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('SiamFC 数据增强效果 - %s (搜索区域)' % seq_name, fontsize=14)

    # 原始
    axes[0].imshow(search_patch.astype(np.uint8))
    axes[0].set_title('原始')
    axes[0].axis('off')

    # RandomStretch
    aug1 = random_stretch(search_patch, 0.05)
    aug1 = center_crop(aug1, INSTANCE_SZ - 8)
    aug1 = random_crop(aug1, INSTANCE_SZ - 16)
    axes[1].imshow(aug1.astype(np.uint8))
    axes[1].set_title('RandomStretch')
    axes[1].axis('off')

    # RandomCrop
    aug2 = random_stretch(search_patch, 0.05)
    aug2 = center_crop(aug2, INSTANCE_SZ - 8)
    aug2 = random_crop(aug2, INSTANCE_SZ - 16)
    axes[2].imshow(aug2.astype(np.uint8))
    axes[2].set_title('RandomCrop')
    axes[2].axis('off')

    # 组合增强
    aug3 = random_stretch(search_patch, 0.05)
    aug3 = center_crop(aug3, INSTANCE_SZ - 8)
    aug3 = random_crop(aug3, INSTANCE_SZ - 16)
    axes[3].imshow(aug3.astype(np.uint8))
    axes[3].set_title('组合增强')
    axes[3].axis('off')

    plt.tight_layout()
    seq_output = os.path.join(OUTPUT_DIR, seq_name)
    if not os.path.exists(seq_output):
        os.makedirs(seq_output)

    img_path = os.path.join(seq_output, 'augment.png')
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('已保存: %s' % img_path)

    # 文字说明
    info = (
        'SiamFC 数据增强效果 - %s\n'
        '========================\n\n'
        '增强方法:\n'
        '1. RandomStretch (随机拉伸): 对图像进行随机尺度缩放，\n'
        '   缩放因子范围 [0.95, 1.05]，模拟目标尺度变化\n'
        '2. RandomCrop (随机裁剪): 在拉伸后的图像上随机裁剪\n'
        '   固定尺寸区域，模拟目标位置偏移\n'
        '3. 组合增强: 上述方法的随机组合\n\n'
        '作用:\n'
        '- 扩充目标外观、尺度、背景的多样性\n'
        '- 降低过拟合，提高模型泛化能力\n'
        '- 增强仅用于训练集，验证/测试集不使用\n\n'
        '实现工具: OpenCV (cv2.resize), 自定义裁剪函数\n'
        '随机种子: 42 (保证可重复)' % seq_name)

    txt_path = os.path.join(seq_output, 'augment_info.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(info)
    print('已保存: %s' % txt_path)


def random_stretch(img, max_stretch=0.05):
    interp = np.random.choice([
        cv2.INTER_LINEAR, cv2.INTER_CUBIC,
        cv2.INTER_AREA, cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4])
    scale = 1.0 + np.random.uniform(-max_stretch, max_stretch)
    out_size = (round(img.shape[1] * scale), round(img.shape[0] * scale))
    return cv2.resize(img, out_size, interpolation=interp)


def center_crop(img, size):
    h, w = img.shape[:2]
    tw, th = size, size
    i = round((h - th) / 2.)
    j = round((w - tw) / 2.)
    npad = max(0, -i, -j)
    if npad > 0:
        avg_color = np.mean(img, axis=(0, 1))
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            cv2.BORDER_CONSTANT, value=avg_color)
        i += npad
        j += npad
    return img[i:i + th, j:j + tw]


def random_crop(img, size):
    h, w = img.shape[:2]
    i = np.random.randint(0, max(1, h - size + 1))
    j = np.random.randint(0, max(1, w - size + 1))
    return img[i:i + size, j:j + size]


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print('生成数据增强可视化...')
    for seq_name in SELECTED_SEQS:
        process_sequence(seq_name)
    print('\n完成! 图片保存在 %s/' % OUTPUT_DIR)


if __name__ == '__main__':
    main()