import os, sys, cv2, numpy as np, csv, time
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')
from models.traditional.color_threshold_tracker import ColorThresholdTracker
from utils.evaluation import Evaluation

test_dir = r'd:/experiment/SingleTrack_Project/data/datasets/GOT_test'
ev = Evaluation()

header = f"{'序列':<25} {'mIoU':<10} {'mCLE':<10} {'成功率':<10} {'FPS':<10} {'丢失次数':<10} {'最大CLE':<10}"
print(header)
print('-' * 85)

all_data = []
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
    t_start = time.time()
    tracker.init(frames[0], gt[0])
    traj = [gt[0]]
    for i in range(1, len(frames)):
        traj.append(tracker.update(frames[i]))
    fps = len(frames) / (time.time() - t_start + 1e-6)
    
    ious = [ev.calculate_iou(traj[i], gt[i]) for i in range(len(traj))]
    cles = [ev.calculate_center_error(traj[i], gt[i]) for i in range(len(traj))]
    
    m_iou = np.mean(ious)
    m_cle = np.mean(cles)
    success = sum(1 for iou in ious if iou >= 0.5) / len(ious)
    lost = tracker.lost_count
    max_cle = max(cles)
    all_data.append((seq_name, m_iou, m_cle, success, fps, lost, max_cle))
    
    print(f'{seq_name:<25} {m_iou:<10.4f} {m_cle:<10.2f} {success:<10.2%} {fps:<10.1f} {lost:<10} {max_cle:<10.2f}')

avg_iou = np.mean([d[1] for d in all_data])
avg_cle = np.mean([d[2] for d in all_data])
avg_success = np.mean([d[3] for d in all_data])
avg_fps = np.mean([d[4] for d in all_data])
avg_lost = np.mean([d[5] for d in all_data])
avg_maxcle = np.mean([d[6] for d in all_data])
print('-' * 85)
print(f'{"平均值":<25} {avg_iou:<10.4f} {avg_cle:<10.2f} {avg_success:<10.2%} {avg_fps:<10.1f} {avg_lost:<10.1f} {avg_maxcle:<10.2f}')

out_dir = r'd:/experiment/SingleTrack_Project/results/traditional'
csv_path = os.path.join(out_dir, 'color_metrics.csv')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['序列','mIoU','mCLE','成功率','FPS','丢失次数','最大CLE'])
    for d in all_data:
        w.writerow([d[0], f'{d[1]:.4f}', f'{d[2]:.2f}', f'{d[3]:.2%}', f'{d[4]:.1f}', d[5], f'{d[6]:.2f}'])
    w.writerow(['平均值', f'{avg_iou:.4f}', f'{avg_cle:.2f}', f'{avg_success:.2%}', f'{avg_fps:.1f}', f'{avg_lost:.1f}', f'{avg_maxcle:.2f}'])
print(f'\n已保存: {csv_path}')