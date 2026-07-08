from __future__ import absolute_import

import os
import glob
import numpy as np
import cv2
from siamfc import TrackerSiamFC


def process_sequence(seq_dir, tracker, output_dir='results_videos'):
    """
    处理单个序列并生成结果视频
    """
    # 获取序列名称
    seq_name = os.path.basename(seq_dir)
    print('Processing sequence: {}'.format(seq_name))

    # 检查序列是否存在
    if not os.path.isdir(seq_dir):
        print('Sequence directory not found: {}'.format(seq_dir))
        return

    # 查找图片文件（支持多种格式）
    img_files = sorted(glob.glob(os.path.join(seq_dir, 'img', '*.jpg')))
    if len(img_files) == 0:
        img_files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))

    if len(img_files) == 0:
        print('No images found in sequence: {}'.format(seq_name))
        return

    # 加载标注文件（逗号分隔格式）
    anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
    if not os.path.isfile(anno_file):
        print('Annotation file not found: {}'.format(anno_file))
        return

    try:
        # 尝试逗号分隔，失败则尝试空格分隔
        try:
            anno = np.loadtxt(anno_file, delimiter=',')
        except:
            anno = np.loadtxt(anno_file)
    except Exception as e:
        print('Error loading annotation file: {} ({})'.format(anno_file, str(e)))
        return

    # 进行跟踪
    try:
        boxes, times = tracker.track(img_files, anno[0], visualize=False)
    except Exception as e:
        print('Tracking failed for sequence {}: {}'.format(seq_name, str(e)))
        return

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成结果视频
    video_path = os.path.join(output_dir, seq_name + '.avi')

    # 读取第一帧获取尺寸
    first_img = cv2.imread(img_files[0])
    if first_img is None:
        print('Error reading first image: {}'.format(img_files[0]))
        return

    height, width = first_img.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    # 确保帧数与标注数一致
    n_frames = min(len(img_files), len(anno))
    if n_frames < len(img_files):
        print('Warning: {} has {} images but {} annotations, using {} frames'.format(
            seq_name, len(img_files), len(anno), n_frames))

    # 写入每一帧
    for i in range(n_frames):
        img_file = img_files[i]
        img = cv2.imread(img_file)
        if img is None:
            continue

        # 绘制真实标注框（红色）
        gt_box = anno[i]
        gt_x, gt_y, gt_w, gt_h = int(gt_box[0]), int(gt_box[1]), int(gt_box[2]), int(gt_box[3])
        cv2.rectangle(img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 0, 255), 2)

        # 绘制跟踪结果框（绿色）
        track_box = boxes[i]
        tr_x, tr_y, tr_w, tr_h = int(track_box[0]), int(track_box[1]), int(track_box[2]), int(track_box[3])
        cv2.rectangle(img, (tr_x, tr_y), (tr_x + tr_w, tr_y + tr_h), (0, 255, 0), 2)

        # 添加帧编号
        cv2.putText(img, 'Frame: {}'.format(i + 1), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加序列名称
        cv2.putText(img, seq_name, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加标注说明
        cv2.putText(img, 'Red: Ground Truth', (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, 'Green: Tracking Result', (10, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        video_writer.write(img)

    video_writer.release()
    print('Video saved: {}'.format(video_path))

    # 计算平均FPS
    avg_fps = len(img_files) / sum(times)
    print('Average FPS: {:.2f}'.format(avg_fps))
    print('')


if __name__ == '__main__':
    # 数据集根目录（与test.py一致）
    dataset_root = r'D:\experiment\SingleTrack_Project\data\datasets\test'

    # 模型路径（你训练了50个epoch的SiamFC模型）
    net_path = 'snapshot_mytrain/siamfc_alexnet_e50.pth'

    # 输出视频目录
    output_dir = 'results_videos'

    # 初始化跟踪器
    print('Loading model from: {}'.format(net_path))
    tracker = TrackerSiamFC(net_path=net_path)
    print('')

    # 获取所有序列目录
    seq_dirs = []
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path):
            seq_dirs.append(item_path)

    print('Found {} sequences in dataset'.format(len(seq_dirs)))
    print('')

    # 批量处理每个序列
    for seq_dir in seq_dirs:
        process_sequence(seq_dir, tracker, output_dir)

    print('All sequences processed!')
    print('Results saved in: {}'.format(output_dir))