from __future__ import absolute_import, print_function, division

import os
import glob
import time
import numpy as np
import cv2

from siamfc import TrackerSiamFC


SELECTED_SEQS = [
    'Basketball', 'Biker', 'BlurCar2', 'Jumping',
    'Coke', 'Girl', 'Jogging', 'Tiger2',
    'CarScale', 'David', 'Dog', 'Singer1',
    'Bolt', 'Dancer', 'Freeman1', 'Skater',
    'Board', 'Deer', 'Liquor', 'Soccer'
]

DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_FILE = r'result\compare\test_results.txt'


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


def test_model(net_path, model_name, seq_names):
    print('\n测试模型: %s (%s)' % (model_name, net_path))
    tracker = TrackerSiamFC(net_path=net_path)
    results = {}
    all_iou, all_cle, all_fps = [], [], []

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

        frame_ious = [compute_iou(pred_boxes[t], anno[t]) for t in range(1, n_frames)]
        frame_cles = [compute_center_error(pred_boxes[t], anno[t]) for t in range(1, n_frames)]

        results[seq_name] = {
            'frames': n_frames,
            'avg_iou': np.mean(frame_ious),
            'success_rate': np.mean(np.array(frame_ious) > 0.5),
            'precision': np.mean(np.array(frame_cles) <= 20),
            'avg_cle': np.mean(frame_cles),
            'avg_fps': n_frames / max(np.sum(times), 1e-6)
        }
        all_iou.extend(frame_ious)
        all_cle.extend(frame_cles)
        all_fps.append(results[seq_name]['avg_fps'])
        print('  %-12s IoU:%.4f Success:%.4f Prec:%.4f CLE:%.1f FPS:%.0f' %
              (seq_name, results[seq_name]['avg_iou'], results[seq_name]['success_rate'],
               results[seq_name]['precision'], results[seq_name]['avg_cle'], results[seq_name]['avg_fps']))

    if len(results) > 0:
        overall = {
            'mean_iou': np.mean(all_iou),
            'mean_success': np.mean([r['success_rate'] for r in results.values()]),
            'mean_precision': np.mean([r['precision'] for r in results.values()]),
            'mean_cle': np.mean(all_cle),
            'mean_fps': np.mean(all_fps),
            'auc': np.trapz(np.sort(all_iou), np.linspace(0, 1, len(all_iou)))
        }
        print('  总体: IoU=%.4f Success=%.4f Prec=%.4f CLE=%.1f FPS=%.0f AUC=%.4f' %
              (overall['mean_iou'], overall['mean_success'], overall['mean_precision'],
               overall['mean_cle'], overall['mean_fps'], overall['auc']))
        return results, overall
    return results, None


def main():
    import os as _os
    _os.makedirs(r'result\compare', exist_ok=True)

    seqs = [s for s in SELECTED_SEQS if _os.path.isdir(_os.path.join(DATASET_ROOT, s))]
    print('测试序列: %d个' % len(seqs))

    results_aug, overall_aug = test_model(
        'snapshot_mytrain/siamfc_alexnet_e30.pth', '有数据增强 (e30)', seqs)

    # 无增强模型
    noaug_path = 'snapshot_noaug/siamfc_alexnet_e40.pth'
    results_noaug, overall_noaug = None, None
    if _os.path.exists(noaug_path):
        results_noaug, overall_noaug = test_model(noaug_path, '无数据增强 (e30)', seqs)
    else:
        print('\n[跳过] 无数据增强模型 e30 不存在')

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('SiamFC 测试结果 (e30最优模型)\n========================\n\n')
        f.write('测试序列: %s\n\n' % ', '.join(seqs))

        for label, res, overall in [('有数据增强 (e30)', results_aug, overall_aug),
                                     ('无数据增强 (e30)', results_noaug, overall_noaug)]:
            if res is None:
                continue
            f.write('--- %s ---\n' % label)
            f.write('%-15s %-6s %-8s %-10s %-10s %-8s\n' %
                    ('序列', '帧数', 'IoU', 'Success', 'Precision', 'CLE'))
            for seq in seqs:
                if seq in res:
                    r = res[seq]
                    f.write('%-15s %-6d %-8.4f %-10.4f %-10.4f %-8.2f\n' %
                            (seq, r['frames'], r['avg_iou'], r['success_rate'],
                             r['precision'], r['avg_cle']))
            if overall:
                f.write('\n总体: IoU=%.4f Success=%.4f Precision=%.4f CLE=%.1f FPS=%.0f AUC=%.4f\n\n' %
                        (overall['mean_iou'], overall['mean_success'], overall['mean_precision'],
                         overall['mean_cle'], overall['mean_fps'], overall['auc']))

    print('\n结果已保存: %s' % OUTPUT_FILE)


if __name__ == '__main__':
    main()