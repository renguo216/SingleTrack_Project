"""
在 GOT_test 全部10个序列上运行边缘轮廓法跟踪器，
替换原有的 edge_contour 结果。
"""
import os
import sys
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')

import cv2
import numpy as np
import csv
import time
from models.traditional.edge_contour_tracker import EdgeContourTracker
from utils.evaluation import Evaluation

test_dir = r'd:/experiment/SingleTrack_Project/data/datasets/GOT_test'
output_base = r'd:/experiment/SingleTrack_Project/results/traditional'

seqs = sorted([s for s in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, s))])
print(f'共{len(seqs)}个序列: {seqs}')

for seq_name in seqs:
    input_dir = os.path.join(test_dir, seq_name)
    save_dir = os.path.join(output_base, seq_name, 'edge_contour')
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载帧
    frames = []
    for f in sorted(os.listdir(input_dir)):
        if f.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(input_dir, f))
            if img is not None:
                frames.append(img)
    
    # 加载GT
    gt = []
    gt_file = os.path.join(input_dir, 'groundtruth.txt')
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            gt.append((int(x), int(y), int(w), int(h)))
    
    print(f'\n[{seq_name}] {len(frames)}帧, 首帧GT={gt[0]}')
    
    # 跟踪
    tracker = EdgeContourTracker()
    tracker.init(frames[0], gt[0])
    
    trajectory = [gt[0]]
    start = time.time()
    for i in range(1, len(frames)):
        pred = tracker.update(frames[i])
        trajectory.append(pred)
    fps = len(frames) / (time.time() - start)
    
    # 计算指标
    evaluator = Evaluation()
    traj_arr = np.array(trajectory)
    unique_x = len(np.unique(traj_arr[:,0]))
    unique_y = len(np.unique(traj_arr[:,1]))
    
    # 保存 trajectory.csv
    csv_path = os.path.join(save_dir, 'trajectory.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['帧号','x','y','w','h','IoU','CLE'])
        for i, (tx, ty, tw, th) in enumerate(trajectory):
            gt_box = gt[i] if i < len(gt) else gt[-1]
            iou = evaluator.calculate_iou((tx, ty, tw, th), gt_box)
            cle = evaluator.calculate_center_error((tx, ty, tw, th), gt_box)
            writer.writerow([i, tx, ty, tw, th, f'{iou:.4f}', f'{cle:.4f}'])
    
    print(f'  X: {traj_arr[:,0].min():.0f}~{traj_arr[:,0].max():.0f} ({unique_x}唯一值)')
    print(f'  Y: {traj_arr[:,1].min():.0f}~{traj_arr[:,1].max():.0f} ({unique_y}唯一值)')
    print(f'  FPS: {fps:.1f}')
    print(f'  轨迹已保存: {csv_path}')

print('\n全部完成!')