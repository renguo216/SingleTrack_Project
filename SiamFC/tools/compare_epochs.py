from __future__ import absolute_import, print_function, division

import os
import glob
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from siamfc import TrackerSiamFC


# 选定的20个代表性序列
SELECTED_SEQS = [
    'Basketball', 'Biker', 'BlurCar2', 'Jumping',
    'Coke', 'Girl', 'Jogging', 'Tiger2',
    'CarScale', 'David', 'Dog', 'Singer1',
    'Bolt', 'Dancer', 'Freeman1', 'Skater',
    'Board', 'Deer', 'Liquor', 'Soccer'
]

DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_FILE = r'result\compare\compare_epochs.txt'
OUTPUT_PNG = r'result\compare\compare_epochs.png'

CHECKPOINTS_DIR = 'snapshot_mytrain'


def get_img_files(seq_dir):
    img_dir = os.path.join(seq_dir, 'img')
    if os.path.isdir(img_dir):
        files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if len(files) > 0:
            return files
    files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    return files


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / max(union, 1e-16)


def compute_center_error(pred_box, gt_box):
    pred_cx = pred_box[0] + pred_box[2] / 2.0
    pred_cy = pred_box[1] + pred_box[3] / 2.0
    gt_cx = gt_box[0] + gt_box[2] / 2.0
    gt_cy = gt_box[1] + gt_box[3] / 2.0
    return np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)


def evaluate_seq(tracker, img_files, anno):
    n = len(img_files)
    pred_boxes = np.zeros((n, 4))
    times = np.zeros(n)
    for f in range(n):
        img = cv2.imread(img_files[f])
        if img is None:
            if f == 0:
                return None
            pred_boxes[f] = pred_boxes[f - 1]
            times[f] = 0
            continue
        begin = time.time()
        if f == 0:
            tracker.init(img, anno[f])
            pred_boxes[f] = anno[f]
        else:
            pred_boxes[f] = tracker.update(img)
        times[f] = time.time() - begin
    return pred_boxes, times


def test_model(net_path, seq_names):
    """测试模型，返回整体指标"""
    tracker = TrackerSiamFC(net_path=net_path)
    all_ious = []
    all_cles = []

    for seq_name in seq_names:
        seq_dir = os.path.join(DATASET_ROOT, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        img_files = get_img_files(seq_dir)
        if len(img_files) == 0:
            continue
        anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
        try:
            try:
                anno = np.loadtxt(anno_file, delimiter=',')
            except:
                anno = np.loadtxt(anno_file)
        except:
            continue
        if len(anno.shape) == 1:
            anno = anno.reshape(1, -1)
        n_frames = min(len(img_files), len(anno))
        try:
            pred_boxes, times = evaluate_seq(tracker, img_files[:n_frames], anno[:n_frames])
            if pred_boxes is None:
                continue
        except:
            continue

        for t in range(1, n_frames):
            all_ious.append(compute_iou(pred_boxes[t], anno[t]))
            all_cles.append(compute_center_error(pred_boxes[t], anno[t]))

    if len(all_ious) == 0:
        return 0, 0, 0

    mean_iou = np.mean(all_ious)
    mean_prec = np.mean(np.array(all_cles) <= 20)
    mean_success = np.mean(np.array(all_ious) > 0.5)
    return mean_iou, mean_prec, mean_success


def main():
    # 检查可用序列
    available = [s for s in SELECTED_SEQS
                 if os.path.isdir(os.path.join(DATASET_ROOT, s))]
    print('可用序列: %d / %d' % (len(available), len(SELECTED_SEQS)))

    # 选取要测试的epoch
    test_epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    results = []
    for epoch in test_epochs:
        net_path = os.path.join(CHECKPOINTS_DIR,
                                'siamfc_alexnet_e%d.pth' % epoch)
        if not os.path.exists(net_path):
            print('[跳过] epoch %d: 模型文件不存在' % epoch)
            continue

        print('测试 epoch %d...' % epoch)
        mean_iou, mean_prec, mean_success = test_model(net_path, available)

        results.append({
            'epoch': epoch,
            'iou': mean_iou,
            'precision': mean_prec,
            'success': mean_success
        })
        print('  IoU: %.4f  Precision: %.4f  Success: %.4f' %
              (mean_iou, mean_prec, mean_success))

    if len(results) == 0:
        print('没有可用的模型进行测试')
        return

    # 保存文本
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('========================================\n')
        f.write('对比实验: 不同训练轮数对性能的影响\n')
        f.write('========================================\n')
        f.write('测试集: 20个OTB2015序列\n')
        f.write('模型: snapshot_mytrain/siamfc_alexnet_e*.pth\n\n')
        f.write('%-10s %-10s %-12s %-12s\n' %
                ('Epoch', 'IoU', 'Precision', 'Success'))
        f.write('-' * 44 + '\n')
        for r in results:
            f.write('%-10d %-10.4f %-12.4f %-12.4f\n' %
                    (r['epoch'], r['iou'], r['precision'], r['success']))

        # 分析收敛
        f.write('\n\n收敛分析:\n')
        best = max(results, key=lambda r: r['iou'])
        f.write('  - 最佳IoU在 epoch %d (IoU=%.4f)\n' % (best['epoch'], best['iou']))
        first = results[0]
        last = results[-1]
        f.write('  - epoch %d -> %d: IoU %.4f -> %.4f (变化 %.4f)\n' %
                (first['epoch'], last['epoch'], first['iou'], last['iou'],
                 last['iou'] - first['iou']))
        f.write('  - epoch %d -> %d: Precision %.4f -> %.4f\n' %
                (first['epoch'], last['epoch'], first['precision'], last['precision']))
        f.write('  - epoch %d -> %d: Success %.4f -> %.4f\n' %
                (first['epoch'], last['epoch'], first['success'], last['success']))

    print('\n对比结果已保存: %s' % OUTPUT_FILE)

    # 画图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('不同训练轮数对SiamFC性能的影响', fontsize=14)

    epochs = [r['epoch'] for r in results]
    ious = [r['iou'] for r in results]
    precs = [r['precision'] for r in results]
    succs = [r['success'] for r in results]

    axes[0].plot(epochs, ious, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('平均IoU')
    axes[0].set_title('IoU vs Epoch')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, precs, 'g-s', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Precision (CLE<=20)')
    axes[1].set_title('Precision vs Epoch')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, succs, 'r-^', linewidth=2, markersize=6)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Success Rate (IoU>0.5)')
    axes[2].set_title('Success vs Epoch')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print('对比图已保存: %s' % OUTPUT_PNG)


if __name__ == '__main__':
    main()