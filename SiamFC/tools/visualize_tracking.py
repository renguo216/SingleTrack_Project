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
OUTPUT_DIR = r'result\tracking_cases'
VIDEO_DIR = r'result\videos'


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


def run_all_sequences():
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
        avg_iou = np.mean(frame_ious)
        results.append({
            'name': seq_name, 'frames': n_frames,
            'pred_boxes': pred_boxes, 'gt_boxes': anno[:n_frames],
            'img_files': img_files[:n_frames],
            'frame_ious': frame_ious, 'frame_cles': frame_cles,
            'avg_iou': avg_iou, 'times': times
        })
        print('  %-12s IoU: %.4f' % (seq_name, avg_iou))
    results.sort(key=lambda r: r['avg_iou'], reverse=True)
    return results


def save_video(result, output_path):
    """保存跟踪视频"""
    n = min(result['frames'], 200)  # 最多200帧
    img0 = cv2.imread(result['img_files'][0])
    h, w = img0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
    for f in range(n):
        img = cv2.imread(result['img_files'][f])
        if img is None:
            continue
        gt = result['gt_boxes'][f]
        pred = result['pred_boxes'][f]
        cv2.rectangle(img, (int(gt[0]), int(gt[1])),
                      (int(gt[0]+gt[2]), int(gt[1]+gt[3])), (0, 0, 255), 2)
        cv2.rectangle(img, (int(pred[0]), int(pred[1])),
                      (int(pred[0]+pred[2]), int(pred[1]+pred[3])), (0, 255, 0), 2)
        cv2.putText(img, 'Frame %d/%d' % (f+1, n), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    print('  视频已保存: %s' % output_path)


def visualize_case(result, case_idx):
    seq_name = result['name']
    n_frames = result['frames']
    pred_boxes = result['pred_boxes']
    gt_boxes = result['gt_boxes']
    img_files = result['img_files']
    frame_ious = result['frame_ious']
    frame_cles = result['frame_cles']

    print('生成案例 %d: %s' % (case_idx, seq_name))

    case_dir = os.path.join(OUTPUT_DIR, 'case%d_%s' % (case_idx, seq_name))
    if not os.path.exists(case_dir):
        os.makedirs(case_dir)

    # ===== 图1: 首帧目标框 + 局部放大 =====
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle('跟踪案例 %d: %s - 首帧与局部放大' % (case_idx, seq_name), fontsize=14)
    img0 = cv2.imread(img_files[0])
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    ax = axes1[0]
    gt0 = gt_boxes[0]
    ax.add_patch(Rectangle((gt0[0], gt0[1]), gt0[2], gt0[3],
                           linewidth=2, edgecolor='red', facecolor='none'))
    ax.imshow(img0_rgb)
    ax.set_title('首帧 (红色=GT, 绿色=预测)')
    ax.axis('off')

    ax = axes1[1]
    # 局部放大预测框+GT框区域
    mid = n_frames // 2
    img_mid = cv2.imread(img_files[mid])
    img_mid_rgb = cv2.cvtColor(img_mid, cv2.COLOR_BGR2RGB)
    pred_mid = pred_boxes[mid]
    gt_mid = gt_boxes[mid]
    cx, cy = (pred_mid[0] + pred_mid[2] / 2), (pred_mid[1] + pred_mid[3] / 2)
    zoom_sz = max(pred_mid[2], pred_mid[3], gt_mid[2], gt_mid[3]) * 3
    x1 = max(0, int(cx - zoom_sz / 2))
    y1 = max(0, int(cy - zoom_sz / 2))
    x2 = min(img_mid_rgb.shape[1], int(cx + zoom_sz / 2))
    y2 = min(img_mid_rgb.shape[0], int(cy + zoom_sz / 2))
    zoom_img = img_mid_rgb[y1:y2, x1:x2].copy()
    pred_local = [pred_mid[0] - x1, pred_mid[1] - y1, pred_mid[2], pred_mid[3]]
    gt_local = [gt_mid[0] - x1, gt_mid[1] - y1, gt_mid[2], gt_mid[3]]
    ax.add_patch(Rectangle((gt_local[0], gt_local[1]), gt_local[2], gt_local[3],
                           linewidth=2, edgecolor='red', facecolor='none'))
    ax.add_patch(Rectangle((pred_local[0], pred_local[1]), pred_local[2], pred_local[3],
                           linewidth=2, edgecolor='lime', facecolor='none'))
    ax.imshow(zoom_img)
    ax.set_title('局部放大 (帧%d, IoU=%.2f)' % (mid + 1, frame_ious[mid - 1] if mid < len(frame_ious) else 0))
    ax.axis('off')
    plt.tight_layout()
    fig1.savefig(os.path.join(case_dir, 'frame1_zoom.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print('  已保存: frame1_zoom.png')

    # ===== 图2: 6个关键帧（上三下三） =====
    n = min(n_frames, 100)
    key_indices = np.linspace(0, n - 1, 3, dtype=int)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('跟踪案例 %d: %s - 关键帧跟踪结果' % (case_idx, seq_name), fontsize=14)
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
        iou_val = frame_ious[fidx - 1] if fidx < len(frame_ious) else 1.0
        ax.imshow(img_rgb)
        ax.set_title('帧%d (IoU=%.2f)' % (fidx + 1, iou_val))
        ax.axis('off')
    plt.tight_layout()
    fig2.savefig(os.path.join(case_dir, 'keyframes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('  已保存: keyframes.png')

    # ===== 图3: IoU折线图（单独一张） =====
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(range(1, len(frame_ious) + 1), frame_ious, 'b-', linewidth=1.5)
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Success阈值(0.5)')
    ax3.set_xlabel('帧')
    ax3.set_ylabel('IoU')
    ax3.set_title('IoU随时间变化 - %s' % seq_name)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(os.path.join(case_dir, 'iou_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print('  已保存: iou_curve.png')

    # ===== 图4: CLE折线图（单独一张） =====
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(range(1, len(frame_cles) + 1), frame_cles, 'r-', linewidth=1.5)
    ax4.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Precision阈值(20px)')
    ax4.set_xlabel('帧')
    ax4.set_ylabel('中心位置误差 (像素)')
    ax4.set_title('中心位置误差随时间变化 - %s' % seq_name)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(os.path.join(case_dir, 'cle_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print('  已保存: cle_curve.png')

    # ===== 图5: Success Plot（单独一张） =====
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    thresholds = np.linspace(0, 1, 100)
    success_rates = [np.mean(np.array(frame_ious) > t) for t in thresholds]
    ax5.plot(thresholds, success_rates, 'b-', linewidth=2)
    ax5.fill_between(thresholds, success_rates, alpha=0.3, color='blue')
    auc = np.trapz(success_rates, thresholds)
    ax5.set_xlabel('IoU阈值')
    ax5.set_ylabel('Success Rate')
    ax5.set_title('Success Plot (AUC=%.4f) - %s' % (auc, seq_name))
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    fig5.savefig(os.path.join(case_dir, 'success_plot.png'), dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print('  已保存: success_plot.png')

    # ===== 文本报告 =====
    good_frames = sum(1 for iou in frame_ious if iou > 0.5)
    report = (
        '跟踪案例 %d: %s\n'
        '========================\n\n'
        '序列信息:\n'
        '  总帧数: %d\n'
        '  平均IoU: %.4f\n'
        '  平均CLE: %.2f 像素\n'
        '  平均FPS: %.1f\n\n'
        '跟踪性能:\n'
        '  成功帧 (IoU>0.5): %d / %d (%.1f%%)\n'
        '  Precision (CLE<=20): %.4f\n'
        '  AUC: %.4f\n' % (
            case_idx, seq_name,
            result['frames'],
            result['avg_iou'], np.mean(frame_cles),
            result['frames'] / max(np.sum(result['times']), 1e-6),
            good_frames, len(frame_ious), 100 * good_frames / max(len(frame_ious), 1),
            np.mean(np.array(frame_cles) <= 20),
            auc))

    txt_path = os.path.join(case_dir, 'report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print('  已保存: report.txt')

    # 保存视频
    video_path = os.path.join(VIDEO_DIR, '%s.avi' % seq_name)
    save_video(result, video_path)


def main():
    for d in [OUTPUT_DIR, VIDEO_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    print('在所有20个序列上运行跟踪...')
    results = run_all_sequences()

    best_5 = results[:5]
    print('\n最佳5个序列:')
    for i, r in enumerate(best_5):
        print('  %d. %s (IoU=%.4f)' % (i + 1, r['name'], r['avg_iou']))

    for i, r in enumerate(best_5):
        visualize_case(r, i + 1)

    print('\n所有可视化案例已保存至: %s' % OUTPUT_DIR)


if __name__ == '__main__':
    main()