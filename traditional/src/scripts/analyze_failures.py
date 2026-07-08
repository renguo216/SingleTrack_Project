"""
分析所有序列的四种方法性能，找出3个最具代表性的失败案例。
输出综合比较表 + 失败案例总结到 results/failure_analysis/
"""
import os
import sys
sys.path.insert(0, 'd:/experiment/SingleTrack_Project')
import cv2
import numpy as np
import csv

from utils.evaluation import Evaluation

test_dir = r'D:/experiment/SingleTrack_Project/data/datasets/GOT_test'
results_dir = r'D:/experiment/SingleTrack_Project/results/traditional'
out_dir = r'D:/experiment/SingleTrack_Project/results/failure_analysis'
os.makedirs(out_dir, exist_ok=True)

methods = {
    'frame_diff': '帧差法',
    'bg_subtract': '背景减除法',
    'color': '颜色阈值法',
    'edge_contour': '边缘轮廓法'
}

ev = Evaluation()

# ========== 收集全部数据 ==========
seq_names = sorted([s for s in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, s))])

all_data = []
for seq in seq_names:
    d = os.path.join(test_dir, seq)
    gt = []
    with open(os.path.join(d, 'groundtruth.txt'), 'r') as f:
        for line in f:
            p = line.strip().split(',')
            x, y, w, h = int(float(p[0])), int(float(p[1])), int(float(p[2])), int(float(p[3]))
            gt.append((x, y, w, h))
    
    for mname, mlabel in methods.items():
        csv_path = os.path.join(results_dir, seq, mname, 'trajectory.csv')
        if not os.path.exists(csv_path):
            continue
        traj = []
        with open(csv_path, 'rb') as f:
            text = f.read().decode('utf-8-sig')
        for line in text.strip().split('\n')[1:]:
            parts = line.strip().split(',')
            x, y, w, h = int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))
            traj.append((x, y, w, h))
        
        n = min(len(traj), len(gt))
        ious = [ev.calculate_iou(traj[i], gt[i]) for i in range(n)]
        cles = [ev.calculate_center_error(traj[i], gt[i]) for i in range(n)]
        success = sum(1 for iou in ious if iou >= 0.5) / n
        
        # 从trajectory.csv最后几列读FPS? 没有，需要从metrics.csv读
        fps = 0.0
        lost = 0
        metrics_f = os.path.join(results_dir, seq, mname, 'metrics.csv')
        if os.path.exists(metrics_f):
            with open(metrics_f, 'rb') as f:
                txt = f.read().decode('utf-8-sig')
            lines = txt.strip().split('\n')
            if len(lines) > 1:
                parts2 = lines[1].strip().split(',')
                if len(parts2) >= 8:
                    try:
                        fps = float(parts2[5])
                        lost = int(float(parts2[6]))
                    except:
                        pass
        
        all_data.append({
            'seq': seq, 'method': mlabel, 'mname': mname,
            'mIoU': np.mean(ious), 'mCLE': np.mean(cles),
            'success': success, 'fps': fps, 'lost': lost,
            'gt_box0': gt[0]
        })

# ========== 生成综合比较表 ==========
print("=" * 90)
print("基础跟踪方法综合比较表")
print("=" * 90)
header = f"{'方法名称':<12} {'关键参数':<30} {'平均CLE':<10} {'平均IoU':<10} {'成功率':<10} {'丢失次数':<10} {'FPS':<10}"
print(header)
print("-" * 90)

method_params = {
    '帧差法': 'diff_thresh=20, kernel=3×3, min_area=100',
    '背景减除法': 'MOG2, history=200, varThresh=30, kernel=5×5',
    '颜色阈值法': 'CamShift, H-S直方图[45×64], kernel=5×5, hist_update=5帧',
    '边缘轮廓法': 'Canny(自适应), dilate(5×5)×2, close(7×7), margin=0.5~4.0'
}

for mname in ['帧差法', '背景减除法', '颜色阈值法', '边缘轮廓法']:
    seq_data = [d for d in all_data if d['method'] == mname]
    avg_iou = np.mean([d['mIoU'] for d in seq_data])
    avg_cle = np.mean([d['mCLE'] for d in seq_data])
    avg_success = np.mean([d['success'] for d in seq_data])
    avg_lost = np.mean([d['lost'] for d in seq_data])
    avg_fps = np.mean([d['fps'] for d in seq_data])
    
    print(f"{mname:<12} {method_params[mname]:<30} {avg_cle:<10.2f} {avg_iou:<10.4f} {avg_success:<10.2%} {avg_lost:<10.1f} {avg_fps:<10.1f}")

# ========== 找3个失败案例 ==========
# 按 mIoU 升序排列，找到最差的 (seq, method) 组合
all_data.sort(key=lambda r: r['mIoU'])
print("\n" + "=" * 90)
print("最差的10个 (seq, method) 组合")
print("=" * 90)
for r in all_data[:10]:
    print(f"  {r['seq']:<25} {r['method']:<10} mIoU={r['mIoU']:.4f} CLE={r['mCLE']:.1f} 成功率={r['success']:.2%} FPS={r['fps']:.1f} 丢失={r['lost']}")

# 挑选3个有代表性的失败案例
failure_cases = []
# Case 1: 颜色阈值法在Val_000146（目标颜色与背景混淆）
# Case 2: 边缘轮廓法在Val_000151（运动模糊）
# Case 3: 帧差法在Val_000162（目标太大几乎覆盖全图）

# 分析每个案例的具体表现
cases_to_analyze = [
    ('GOT-10k_Val_000146', '颜色阈值法', '目标颜色与背景混淆'),
    ('GOT-10k_Val_000151', '边缘轮廓法', '大目标部分运动，边缘信息不足'),
    ('GOT-10k_Val_000162', '帧差法', '目标尺寸极大，帧差法难以检测内部变化'),
]

print("\n" + "=" * 90)
print("3个失败案例详细分析")
print("=" * 90)

for seq_name, method_name, issue in cases_to_analyze:
    # 获取该序列该方法的metrics
    d = os.path.join(test_dir, seq_name)
    gt = []
    with open(os.path.join(d, 'groundtruth.txt'), 'r') as f:
        for line in f:
            p = line.strip().split(',')
            x, y, w, h = int(float(p[0])), int(float(p[1])), int(float(p[2])), int(float(p[3]))
            gt.append((x, y, w, h))
    
    # 获取帧数
    frames_cnt = len([f for f in os.listdir(d) if f.endswith(('.jpg','.png'))])
    
    # 找对应数据
    entry = None
    for e in all_data:
        if e['seq'] == seq_name and e['method'] == method_name:
            entry = e
            break
    
    if entry is None:
        continue
    
    print(f"\n案例: {seq_name} - {method_name}")
    print(f"  帧数: {frames_cnt}")
    print(f"  首帧GT框: {entry['gt_box0']}")
    print(f"  mIoU: {entry['mIoU']:.4f}")
    print(f"  mCLE: {entry['mCLE']:.1f}px")
    print(f"  成功率: {entry['success']:.2%}")
    print(f"  FPS: {entry['fps']:.1f}")
    print(f"  丢失次数: {entry['lost']}")
    print(f"  失败原因: {issue}")
    
    # 其他三种方法在该序列的表现对比
    for mname2 in ['帧差法', '背景减除法', '颜色阈值法', '边缘轮廓法']:
        if mname2 == method_name:
            continue
        for e2 in all_data:
            if e2['seq'] == seq_name and e2['method'] == mname2:
                print(f"  {mname2}: mIoU={e2['mIoU']:.4f} CLE={e2['mCLE']:.1f}")

# ========== 保存CSV ==========
csv_path = os.path.join(out_dir, 'comprehensive_comparison.csv')
with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['序号', '序列', '方法', 'mIoU', 'mCLE', '成功率', 'FPS', '丢失次数'])
    for i, r in enumerate(all_data):
        w.writerow([i+1, r['seq'], r['method'],
                    f"{r['mIoU']:.4f}", f"{r['mCLE']:.2f}",
                    f"{r['success']:.2%}", f"{r['fps']:.1f}", r['lost']])

print(f"\n综合比较表已保存: {csv_path}")