"""
测试修改后的颜色阈值法跟踪器在所有10个序列上的表现。
"""
import os
import sys
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')
import cv2
import numpy as np
import csv
from models.traditional.color_threshold_tracker import ColorThresholdTracker
from utils.evaluation import Evaluation

test_dir = r'd:/experiment/SingleTrack_Project/data/datasets/GOT_test'
out_base = r'd:/experiment/SingleTrack_Project/results/traditional'
ev = Evaluation()

results = []
for seq_name in sorted(os.listdir(test_dir)):
    d = os.path.join(test_dir, seq_name)
    if not os.path.isdir(d):
        continue
    
    frames = []
    for f in sorted(os.listdir(d)):
        if f.endswith(('.jpg','.png')):
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                frames.append(img)
    
    gt = []
    with open(os.path.join(d, 'groundtruth.txt'), 'r') as f:
        for line in f:
            p = line.strip().split(',')
            gt.append((int(float(p[0])),int(float(p[1])),int(float(p[2])),int(float(p[3]))))
    
    tracker = ColorThresholdTracker()
    tracker.init(frames[0], gt[0])
    traj = [gt[0]]
    for i in range(1, len(frames)):
        traj.append(tracker.update(frames[i]))
    
    ious = [ev.calculate_iou(traj[i], gt[i]) for i in range(len(traj))]
    tr = np.array(traj)
    x_unique = len(np.unique(tr[:,0]))
    y_unique = len(np.unique(tr[:,1]))
    m_iou = np.mean(ious)
    
    print(f'[{seq_name}] X:{x_unique} Y:{y_unique} mIoU={m_iou:.4f}')
    results.append((seq_name, x_unique, y_unique, m_iou))

print('\n=== 排序（按mIoU）===')
results.sort(key=lambda r: -r[3])
for r in results:
    print(f'  {r[0]}: X={r[1]}, Y={r[2]}, mIoU={r[3]:.4f}')