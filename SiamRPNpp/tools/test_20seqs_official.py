import argparse
import os, sys, time, cv2, numpy as np, torch, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain

SELECTED_SEQS = [
    'Basketball', 'Biker', 'BlurCar2', 'Jumping',
    'Coke', 'Girl', 'Jogging', 'Tiger2',
    'CarScale', 'David', 'Dog', 'Singer1',
    'Bolt', 'Dancer', 'Freeman1', 'Skater',
    'Board', 'Deer', 'Liquor', 'Soccer'
]

DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_DIR = r'D:\experiment\pysot-master\results\OTB100\SiamRPNpp'
OUTPUT_TXT = r'D:\experiment\pysot-master\results\compare_rpnpp\test_results.txt'


def get_img_files(seq_dir):
    img_dir = os.path.join(seq_dir, 'img')
    if os.path.isdir(img_dir):
        files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if files:
            return files
    return sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))


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


def main():
    # 加载官方预训练模型配置
    cfg.merge_from_file('experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
    model = ModelBuilder()
    model = load_pretrain(model, 'experiments/siamrpn_r50_l234_dwxcorr/model.pth').cuda().eval()
    tracker = build_tracker(model)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    results = {}
    all_iou, all_cle, all_fps = [], [], []

    print('SiamRPN++ (官方预训练) 测试结果')
    print('=' * 70)

    for seq_name in SELECTED_SEQS:
        seq_dir = os.path.join(DATASET_ROOT, seq_name)
        if not os.path.isdir(seq_dir):
            print(f'  [跳过] {seq_name}')
            continue

        img_files = get_img_files(seq_dir)
        if not img_files:
            continue

        anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
        try:
            anno = np.loadtxt(anno_file, delimiter=',')
        except:
            try:
                anno = np.loadtxt(anno_file)
            except:
                continue
        if len(anno.shape) == 1:
            anno = anno.reshape(1, -1)

        # 重置tracker（每个视频重新创建）
        model_v = ModelBuilder()
        model_v = load_pretrain(model_v, 'experiments/siamrpn_r50_l234_dwxcorr/model.pth').cuda().eval()
        tracker_v = build_tracker(model_v)

        n_frames = min(len(img_files), len(anno))
        pred_boxes = np.zeros((n_frames, 4))
        times = np.zeros(n_frames)

        for f in range(n_frames):
            img = cv2.imread(img_files[f])
            if img is None:
                if f == 0:
                    break
                pred_boxes[f] = pred_boxes[f - 1]
                continue

            begin = time.time()
            if f == 0:
                cx, cy, w, h = get_axis_aligned_bbox(anno[f])
                init_box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker_v.init(img, init_box)
                pred_boxes[f] = anno[f]
            else:
                outputs = tracker_v.track(img)
                pred_boxes[f] = outputs['bbox']
            times[f] = time.time() - begin

        # 保存跟踪结果
        result_path = os.path.join(OUTPUT_DIR, f'{seq_name.lower()}.txt')
        with open(result_path, 'w') as f:
            for box in pred_boxes:
                f.write(f'{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}\n')

        # 指标（从第2帧开始）
        frame_ious = [compute_iou(pred_boxes[t], anno[t]) for t in range(1, n_frames)]
        frame_cles = [compute_center_error(pred_boxes[t], anno[t]) for t in range(1, n_frames)]

        iou = np.mean(frame_ious)
        success = np.mean(np.array(frame_ious) > 0.5)
        precision = np.mean(np.array(frame_cles) <= 20)
        cle = np.mean(frame_cles)
        fps = n_frames / max(np.sum(times), 1e-6)

        results[seq_name] = {
            'frames': n_frames, 'avg_iou': iou,
            'success_rate': success, 'precision': precision,
            'avg_cle': cle, 'avg_fps': fps
        }
        all_iou.extend(frame_ious)
        all_cle.extend(frame_cles)
        all_fps.append(fps)

        print(f'  {seq_name:<12s} IoU:{iou:.4f} Success:{success:.4f} Prec:{precision:.4f} CLE:{cle:.1f} FPS:{fps:.0f}')

    # 总体
    if results:
        overall_iou = np.mean(all_iou)
        overall_success = np.mean([r['success_rate'] for r in results.values()])
        overall_precision = np.mean([r['precision'] for r in results.values()])
        overall_cle = np.mean(all_cle)
        overall_fps = np.mean(all_fps)
        sorted_iou = np.sort(all_iou)
        auc = np.trapz(sorted_iou, np.linspace(0, 1, len(sorted_iou))) if len(sorted_iou) > 1 else 0

        print('=' * 70)
        print(f'  总体: IoU={overall_iou:.4f} Success={overall_success:.4f} Prec={overall_precision:.4f} CLE={overall_cle:.1f} FPS={overall_fps:.0f} AUC={auc:.4f}')

        with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
            f.write('SiamRPN++ 测试结果 (官方预训练模型)\n')
            f.write('=' * 60 + '\n\n')
            f.write('测试序列: ' + ', '.join(SELECTED_SEQS) + '\n\n')
            f.write(f'{"序列":<15s} {"帧数":<6s} {"IoU":<8s} {"Success":<10s} {"Precision":<10s} {"CLE":<8s}\n')
            for seq in SELECTED_SEQS:
                if seq in results:
                    r = results[seq]
                    f.write(f'{seq:<15s} {r["frames"]:<6d} {r["avg_iou"]:<8.4f} {r["success_rate"]:<10.4f} {r["precision"]:<10.4f} {r["avg_cle"]:<8.2f}\n')
            f.write(f'\n总体: IoU={overall_iou:.4f} Success={overall_success:.4f} Precision={overall_precision:.4f} CLE={overall_cle:.1f} FPS={overall_fps:.0f} AUC={auc:.4f}\n')

        print(f'\n结果保存至: {OUTPUT_DIR}')
        print(f'指标保存至: {OUTPUT_TXT}')


if __name__ == '__main__':
    main()