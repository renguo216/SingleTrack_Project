from __future__ import absolute_import, print_function, division

import os
import glob
import cv2
import numpy as np


DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_FILE = 'dataset_info.txt'


def get_img_files(seq_dir):
    """获取序列所有图片文件"""
    # 先尝试 img/ 子目录
    img_dir = os.path.join(seq_dir, 'img')
    if os.path.isdir(img_dir):
        files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if len(files) > 0:
            return files
        files = sorted(glob.glob(os.path.join(img_dir, '*.bmp')))
        if len(files) > 0:
            return files
    # 直接目录下
    files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    if len(files) > 0:
        return files
    files = sorted(glob.glob(os.path.join(seq_dir, '*.bmp')))
    return files


def analyze_dataset(root_dir):
    """分析 OTB2015 数据集基本信息"""
    seq_names = sorted([d for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d))])

    info = {
        'num_seqs': len(seq_names),
        'frames_range': [99999, 0],
        'img_sizes': set(),
        'anno_format': 'groundtruth_rect.txt',
        'seq_details': []
    }

    total_frames = 0
    for seq_name in seq_names:
        seq_dir = os.path.join(root_dir, seq_name)
        img_files = get_img_files(seq_dir)
        n_frames = len(img_files)
        total_frames += n_frames
        if n_frames < info['frames_range'][0]:
            info['frames_range'][0] = n_frames
        if n_frames > info['frames_range'][1]:
            info['frames_range'][1] = n_frames

        # 读取标注
        anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
        try:
            try:
                anno = np.loadtxt(anno_file, delimiter=',')
            except:
                anno = np.loadtxt(anno_file)
        except:
            anno = None

        # 图片尺寸
        w, h = 0, 0
        if n_frames > 0:
            first_img = cv2.imread(img_files[0])
            if first_img is not None:
                h, w = first_img.shape[:2]
                info['img_sizes'].add((w, h))

        anno_shape_str = str(anno.shape) if anno is not None else 'N/A'
        info['seq_details'].append({
            'name': seq_name,
            'frames': n_frames,
            'img_size': '%dx%d' % (w, h),
            'anno_shape': anno_shape_str
        })

    info['total_frames'] = total_frames
    info['avg_frames'] = total_frames // max(1, len(seq_names))
    return info, seq_names


def show_typical_sequences(seq_names, base_dir):
    """展示2组典型序列的首帧和关键帧（追加到输出文件）"""
    examples = ['Basketball', 'Girl']
    available = [s for s in examples if s in seq_names]
    if len(available) < 2:
        available = seq_names[:2]

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write('\n\n===== 典型序列展示 =====\n')
        for seq_name in available:
            seq_dir = os.path.join(base_dir, seq_name)
            img_files = get_img_files(seq_dir)

            anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
            try:
                try:
                    anno = np.loadtxt(anno_file, delimiter=',')
                except:
                    anno = np.loadtxt(anno_file)
            except:
                anno = None

            f.write('\n序列: %s (共%d帧)\n' % (seq_name, len(img_files)))
            if anno is not None:
                f.write('  标注文件格式: %s\n' % str(anno.shape))
                f.write('  首帧目标框: [x=%.1f, y=%.1f, w=%.1f, h=%.1f]\n' % tuple(anno[0]))

            # 选取关键帧: 首帧、中间帧、末帧
            key_frames = [0]
            if len(img_files) > 2:
                key_frames.append(len(img_files) // 2)
            if len(img_files) > 1:
                key_frames.append(len(img_files) - 1)

            for kf in key_frames:
                img = cv2.imread(img_files[kf])
                if img is None:
                    continue
                h, w = img.shape[:2]
                if anno is not None:
                    f.write('  帧%d: 尺寸 %dx%d, 目标框 [x=%.1f, y=%.1f, w=%.1f, h=%.1f]\n' %
                            (kf + 1, w, h,
                             anno[kf][0], anno[kf][1], anno[kf][2], anno[kf][3]))
                else:
                    f.write('  帧%d: 尺寸 %dx%d\n' % (kf + 1, w, h))
            f.write('\n')


def main():
    print('正在分析OTB2015数据集...')
    info, seq_names = analyze_dataset(DATASET_ROOT)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('========================================\n')
        f.write('OTB2015 数据集基本信息\n')
        f.write('========================================\n')
        f.write('数据集来源: OTB2015 (Object Tracking Benchmark 2015)\n')
        f.write('序列总数: %d\n' % info['num_seqs'])
        f.write('总帧数: %d\n' % info['total_frames'])
        f.write('帧数范围: %d ~ %d\n' % (info['frames_range'][0], info['frames_range'][1]))
        f.write('平均帧数: %d\n' % info['avg_frames'])
        f.write('标注文件格式: %s\n' % info['anno_format'])
        f.write('标注内容: [left_x, top_y, width, height]\n')
        f.write('图片格式: JPG\n')
        f.write('颜色空间: RGB\n')
        f.write('原始尺寸分布: %s\n' % ', '.join(['%dx%d' % s for s in sorted(info['img_sizes'])]))

        f.write('\n\n========================================\n')
        f.write('所有序列详情\n')
        f.write('========================================\n')
        for d in info['seq_details']:
            f.write('  %-15s 帧数:%-4d 尺寸:%-10s 标注: %s\n' %
                    (d['name'], d['frames'], d['img_size'], d['anno_shape']))

    show_typical_sequences(seq_names, DATASET_ROOT)

    print('\n数据集分析完成!')
    print('  序列总数: %d' % info['num_seqs'])
    print('  总帧数: %d' % info['total_frames'])
    print('  帧数范围: %d ~ %d' % (info['frames_range'][0], info['frames_range'][1]))
    print('文件已保存: %s' % OUTPUT_FILE)


if __name__ == '__main__':
    main()