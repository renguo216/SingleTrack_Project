"""
在全部10个序列上运行颜色阈值法，更新trajectory.csv并生成视频。
"""
import os
import sys
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')
import cv2
import numpy as np
import csv
import time
import shutil
from models.traditional.color_threshold_tracker import ColorThresholdTracker
from utils.evaluation import Evaluation

test_dir = r'd:/experiment/SingleTrack_Project/data/datasets/GOT_test'
out_base = r'd:/experiment/SingleTrack_Project/results/traditional'
ev = Evaluation()

for seq_name in sorted(os.listdir(test_dir)):
    d = os.path.join(test_dir, seq_name)
    if not os.path.isdir(d):
        continue
    
    print(f'\n[{seq_name}]')
    
    # 加载
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
    
    save_dir = os.path.join(out_base, seq_name, 'color')
    os.makedirs(save_dir, exist_ok=True)
    
    # 跟踪
    tracker = ColorThresholdTracker()
    tracker.init(frames[0], gt[0])
    traj = [gt[0]]
    for i in range(1, len(frames)):
        traj.append(tracker.update(frames[i]))
    
    # 统计
    ious = [ev.calculate_iou(traj[i], gt[i]) for i in range(len(traj))]
    tr = np.array(traj)
    x_u = len(np.unique(tr[:,0]))
    y_u = len(np.unique(tr[:,1]))
    moved = sum(1 for i in range(1, len(traj)) if traj[i] != traj[0])
    print(f'  X:{x_u} Y:{y_u} 移动:{moved}/{len(frames)-1} mIoU={np.mean(ious):.4f}')
    
    # 保存 trajectory.csv
    with open(os.path.join(save_dir, 'trajectory.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['帧号','x','y','w','h','IoU','CLE'])
        for i, (tx,ty,tw,th) in enumerate(traj):
            iou = ev.calculate_iou((tx,ty,tw,th), gt[i])
            cle = ev.calculate_center_error((tx,ty,tw,th), gt[i])
            w.writerow([i, tx, ty, tw, th, f'{iou:.4f}', f'{cle:.4f}'])
    
    # 生成视频
    H, W = frames[0].shape[:2]
    video_path = os.path.join(save_dir, f'{seq_name}_color.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (W, H))
    
    for i, frame in enumerate(frames):
        vis = frame.copy()
        # GT 黄色
        if i < len(gt):
            x, y, w, h = gt[i]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 255), 3)
            cv2.putText(vis, 'GT', (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # 预测 红色
        if i < len(traj):
            x, y, w, h = traj[i]
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(vis, 'Color', (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, f'Frame {i+1}/{len(frames)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(vis)
    
    out.release()
    print(f'  视频已保存')

print('\n全部完成！')