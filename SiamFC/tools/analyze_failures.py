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
from matplotlib.patches import Rectangle

from siamfc import TrackerSiamFC


SELECTED_SEQS = [
    'Basketball', 'Biker', 'BlurCar2', 'Jumping',
    'Coke', 'Girl', 'Jogging', 'Tiger2',
    'CarScale', 'David', 'Dog', 'Singer1',
    'Bolt', 'Dancer', 'Freeman1', 'Skater',
    'Board', 'Deer', 'Liquor', 'Soccer'
]

DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
MODEL_PATH = 'snapshot_mytrain/siamfc_alexnet_e30.pth'
OUTPUT_DIR = r'result\failure_cases'


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


def run_and_rank():
    tracker = TrackerSiamFC(net_path=MODEL_PATH)
    results = []
    for seq_name in SELECTED_SEQS:
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
        pred_boxes = np.zeros((n_frames, 4))
        times = np.zeros(n_frames)
        success = True
        for f in range(n_frames):
            img = cv2.imread(img_files[f])
            if img is None:
                if f == 0:
                    success = False
                    break
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
        if not success:
            continue
        frame_ious = [compute_iou(pred_boxes[t], anno[t]) for t in range(1, n_frames)]
        frame_cles = [compute_center_error(pred_boxes[t], anno[t]) for t in range(1, n_frames)]
        results.append({
            'name': seq_name, 'frames': n_frames,
            'pred_boxes': pred_boxes, 'gt_boxes': anno[:n_frames],
            'img_files': img_files[:n_frames],
            'frame_ious': frame_ious, 'frame_cles': frame_cles,
            'avg_iou': np.mean(frame_ious), 'times': times
        })
        print('  %-12s IoU: %.4f' % (seq_name, np.mean(frame_ious)))
    results.sort(key=lambda r: r['avg_iou'])
    return results


def analyze_failure(result, case_idx):
    seq_name = result['name']
    n_frames = result['frames']
    pred_boxes = result['pred_boxes']
    gt_boxes = result['gt_boxes']
    img_files = result['img_files']
    frame_ious = result['frame_ious']
    frame_cles = result['frame_cles']

    print('生成失败案例 %d: %s' % (case_idx, seq_name))

    case_dir = os.path.join(OUTPUT_DIR, 'case%d_%s' % (case_idx, seq_name))
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    # 失败帧分析
    fail_frames = [i + 1 for i, iou in enumerate(frame_ious) if iou < 0.3]

    # 选取展示帧
    display_indices = [0]
    for i, iou in enumerate(frame_ious):
        if iou < 0.3:
            display_indices.append(i + 1)
            break
    worst_iou_idx = np.argmin(frame_ious)
    if worst_iou_idx + 1 not in display_indices:
        display_indices.append(worst_iou_idx + 1)
    display_indices = sorted(set(display_indices))[:5]

    # 失败原因分析
    failure_reasons = []
    if 'Jogging' in seq_name or 'Girl' in seq_name or 'Coke' in seq_name or 'Tiger2' in seq_name:
        failure_reasons.append('严重遮挡 (Severe Occlusion)')
    if 'Biker' in seq_name or 'BlurCar2' in seq_name or 'Jumping' in seq_name:
        failure_reasons.append('快速运动/运动模糊 (Fast Motion / Motion Blur)')
    if 'Bolt' in seq_name or 'Dancer' in seq_name or 'Freeman1' in seq_name:
        failure_reasons.append('目标形变/非刚性形变 (Deformation)')
    if 'Board' in seq_name or 'Deer' in seq_name or 'Liquor' in seq_name or 'Soccer' in seq_name:
        failure_reasons.append('背景相似干扰 (Background Clutter)')
    if 'CarScale' in seq_name or 'Dog' in seq_name or 'Singer1' in seq_name:
        failure_reasons.append('尺度剧烈变化 (Scale Variation)')
    if not failure_reasons:
        failure_reasons.append('综合因素导致跟踪漂移 (Tracking Drift)')

    total_fail = len(fail_frames)
    fail_ratio = total_fail / max(n_frames, 1)

    # ===== 图1: 首帧 + 失败帧局部放大 =====
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('失败案例 %d: %s - 首帧与跟踪丢失对比' % (case_idx, seq_name), fontsize=14)

    # 首帧
    img0 = cv2.imread(img_files[0])
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    ax = axes1[0]
    gt0 = gt_boxes[0]
    ax.add_patch(Rectangle((gt0[0], gt0[1]), gt0[2], gt0[3],
                           linewidth=2, edgecolor='red', facecolor='none'))
    ax.imshow(img0_rgb)
    ax.set_title('首帧目标框')
    ax.axis('off')

    # 失败帧局部放大
    ax = axes1[1]
    if len(display_indices) > 1:
        fail_fidx = display_indices[1]
        img_fail = cv2.imread(img_files[fail_fidx])
        img_fail_rgb = cv2.cvtColor(img_fail, cv2.COLOR_BGR2RGB)
        pred_fail = pred_boxes[fail_fidx]
        gt_fail = gt_boxes[fail_fidx]
        cx, cy = (pred_fail[0] + pred_fail[2] / 2), (pred_fail[1] + pred_fail[3] / 2)
        zoom_sz = max(pred_fail[2], pred_fail[3], gt_fail[2], gt_fail[3]) * 3
        x1 = max(0, int(cx - zoom_sz / 2))
        y1 = max(0, int(cy - zoom_sz / 2))
        x2 = min(img_fail_rgb.shape[1], int(cx + zoom_sz / 2))
        y2 = min(img_fail_rgb.shape[0], int(cy + zoom_sz / 2))
        zoom_img = img_fail_rgb[y1:y2, x1:x2].copy()
        pred_local = [pred_fail[0] - x1, pred_fail[1] - y1, pred_fail[2], pred_fail[3]]
        gt_local = [gt_fail[0] - x1, gt_fail[1] - y1, gt_fail[2], gt_fail[3]]
        ax.add_patch(Rectangle((gt_local[0], gt_local[1]), gt_local[2], gt_local[3],
                               linewidth=2, edgecolor='red', facecolor='none', label='GT'))
        ax.add_patch(Rectangle((pred_local[0], pred_local[1]), pred_local[2], pred_local[3],
                               linewidth=2, edgecolor='lime', facecolor='none', label='Pred'))
        ax.imshow(zoom_img)
        ax.set_title('跟踪丢失局部放大 (帧%d, IoU=%.2f)' % (fail_fidx + 1, frame_ious[fail_fidx - 1]))
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, '无典型丢失帧', ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')
    plt.tight_layout()
    fig1.savefig(os.path.join(case_dir, 'frame1_fail_zoom.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print('  已保存: frame1_fail_zoom.png')

    # ===== 图2: 6个关键帧 =====
    n = min(n_frames, 100)
    key_indices = np.linspace(0, n - 1, 3, dtype=int)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('失败案例 %d: %s - 关键帧 (红色标注丢失)' % (case_idx, seq_name), fontsize=14)
    for i, fidx in enumerate(key_indices):
        ax = axes2[i]
        img = cv2.imread(img_files[fidx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt = gt_boxes[fidx]
        pred = pred_boxes[fidx]
        ax.add_patch(Rectangle((gt[0], gt[1]), gt[2], gt[3],
                               linewidth=2, edgecolor='red', facecolor='none'))
        ax.add_patch(Rectangle((pred[0], pred[1]), pred[2], pred[3],
                               linewidth=2, edgecolor='lime', facecolor='none'))
        iou_val = frame_ious[fidx - 1] if fidx < len(frame_ious) else 0
        title = '帧%d (IoU=%.2f)' % (fidx + 1, iou_val)
        ax.set_title(title, color='red' if iou_val < 0.3 else 'black')
        ax.imshow(img_rgb)
        ax.axis('off')
    plt.tight_layout()
    fig2.savefig(os.path.join(case_dir, 'keyframes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('  已保存: keyframes.png')

    # ===== 图3: IoU折线图（标注失败区域） =====
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(range(1, len(frame_ious) + 1), frame_ious, 'b-', linewidth=1.5)
    ax3.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Success阈值(0.5)')
    ax3.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='严重丢失阈值(0.3)')
    for ff in fail_frames:
        ax3.axvspan(max(1, ff - 2), min(len(frame_ious), ff + 2), alpha=0.2, color='red')
    ax3.set_xlabel('帧')
    ax3.set_ylabel('IoU')
    ax3.set_title('IoU曲线 (红色区域=跟踪丢失) - %s' % seq_name)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(case_dir, 'iou_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print('  已保存: iou_curve.png')

    # ===== 图4: CLE折线图 =====
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(range(1, len(frame_cles) + 1), frame_cles, 'r-', linewidth=1.5)
    ax4.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Precision阈值(20px)')
    for ff in fail_frames:
        ax4.axvspan(max(1, ff - 2), min(len(frame_cles), ff + 2), alpha=0.2, color='red')
    ax4.set_xlabel('帧')
    ax4.set_ylabel('中心位置误差 (像素)')
    ax4.set_title('中心位置误差曲线 - %s' % seq_name)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(os.path.join(case_dir, 'cle_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print('  已保存: cle_curve.png')

    # ===== 文本报告 =====
    report = (
        '失败案例 %d: %s\n'
        '========================\n\n'
        '一、基本统计\n'
        '  总帧数: %d\n'
        '  平均IoU: %.4f\n'
        '  平均CLE: %.2f 像素\n'
        '  失败帧数 (IoU<0.3): %d (%.1f%%)\n'
        '  最大CLE: %.1f 像素\n\n'
        '二、失败原因分析\n' % (case_idx, seq_name, n_frames,
                             result['avg_iou'], np.mean(frame_cles),
                             total_fail, 100 * fail_ratio, np.max(frame_cles)))
    for j, reason in enumerate(failure_reasons):
        report += '  %d. %s\n' % (j + 1, reason)

    report += (
        '\n三、改进思路\n'
        '  1. 引入模板更新策略：定期或根据置信度更新模板\n'
        '  2. 增加在线微调：跟踪过程中对部分帧梯度回传微调模型\n'
        '  3. 多尺度搜索：使用更精细的尺度金字塔搜索\n'
        '  4. 增加遮挡样本的数据增强\n'
        '  5. 使用更深骨干网络 (如ResNet代替AlexNet)\n')

    txt_path = os.path.join(case_dir, 'analysis.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print('  已保存: analysis.txt')

    return {'name': seq_name, 'avg_iou': result['avg_iou'], 'reasons': failure_reasons}


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print('运行所有20个序列，排序取最差3个...')
    results = run_and_rank()
    worst_3 = results[:3]

    print('\n最差3个序列:')
    for i, r in enumerate(worst_3):
        print('  %d. %s (IoU=%.4f)' % (i + 1, r['name'], r['avg_iou']))

    analyses = []
    for i, r in enumerate(worst_3):
        analyses.append(analyze_failure(r, i + 1))

    # 汇总报告
    summary = '失败分析汇总\n========================\n\n'
    for a in analyses:
        summary += '- %s: IoU=%.4f, 原因: %s\n' % (a['name'], a['avg_iou'], ', '.join(a['reasons']))
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print('\n失败分析汇总已保存: %s' % summary_path)


if __name__ == '__main__':
    main()