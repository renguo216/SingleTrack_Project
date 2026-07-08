from __future__ import absolute_import, print_function, division

import os, glob, time, numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from siamfc import TrackerSiamFC


SELECTED_SEQS = [
    'Basketball', 'Biker', 'BlurCar2', 'Jumping',
    'Coke', 'Girl', 'Jogging', 'Tiger2',
    'CarScale', 'David', 'Dog', 'Singer1',
    'Bolt', 'Dancer', 'Freeman1', 'Skater',
    'Board', 'Deer', 'Liquor', 'Soccer'
]
DATASET_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_FILE = r'result\compare\augmentation_comparison.txt'
OUTPUT_PNG = r'result\compare\augmentation_comparison.png'


def get_img_files(seq_dir):
    img_dir = os.path.join(seq_dir, 'img')
    if os.path.isdir(img_dir):
        files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if len(files) > 0: return files
    return sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))

def compute_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / max(union, 1e-16)

def compute_cle(p, g):
    return np.sqrt((p[0]+p[2]/2-g[0]-g[2]/2)**2 + (p[1]+p[3]/2-g[1]-g[3]/2)**2)

def evaluate(tracker, img_files, anno):
    n = len(img_files)
    pred = np.zeros((n, 4))
    for f in range(n):
        img = cv2.imread(img_files[f])
        if img is None:
            if f == 0: return None
            pred[f] = pred[f-1]
            continue
        if f == 0:
            tracker.init(img, anno[f])
            pred[f] = anno[f]
        else:
            pred[f] = tracker.update(img)
    return pred

def test_model(net_path, seq_names):
    print('  模型: %s' % net_path)
    try:
        tracker = TrackerSiamFC(net_path=net_path)
    except:
        return {}
    results = {}
    for s in seq_names:
        sd = os.path.join(DATASET_ROOT, s)
        if not os.path.isdir(sd): continue
        imgs = get_img_files(sd)
        if not imgs: continue
        try:
            anno = np.loadtxt(os.path.join(sd, 'groundtruth_rect.txt'))
        except:
            try:
                anno = np.loadtxt(os.path.join(sd, 'groundtruth_rect.txt'), delimiter=',')
            except:
                continue
        if len(anno.shape) == 1: anno = anno.reshape(1, -1)
        n = min(len(imgs), len(anno))
        pred = evaluate(tracker, imgs[:n], anno[:n])
        if pred is None: continue
        ious = [compute_iou(pred[t], anno[t]) for t in range(1, n)]
        cles = [compute_cle(pred[t], anno[t]) for t in range(1, n)]
        if ious:
            results[s] = {'iou': np.mean(ious), 'cle': np.mean(cles),
                          'success': np.mean(np.array(ious) > 0.5),
                          'prec': np.mean(np.array(cles) <= 20)}
        print('    %s done' % s)
    return results

def main():
    os.makedirs(r'result\compare', exist_ok=True)
    seqs = [s for s in SELECTED_SEQS if os.path.isdir(os.path.join(DATASET_ROOT, s))]
    print('可用序列: %d个' % len(seqs))

    r_aug = test_model('snapshot_mytrain/siamfc_alexnet_e30.pth', seqs)
    
    noaug = 'snapshot_noaug/siamfc_alexnet_e40.pth'
    r_noaug = {}
    if os.path.exists(noaug):
        r_noaug = test_model(noaug, seqs)
    else:
        print('\n[注意] 无增强模型 e30 不存在')

    # 写文本
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('对比实验: 有数据增强 vs 无数据增强\n')
        f.write('控制变量: 同AlexNetV1+SiamFC, epoch=50→选e30为最优\n')
        f.write('变量: 有增强(RandomStretch+RandomCrop) vs 无增强(仅CenterCrop)\n\n')
        f.write('%-15s %-10s %-10s %-10s %-10s %-10s\n' %
                ('序列', 'IoU(增)', 'IoU(无)', 'Prec(增)', 'Prec(无)', 'FPS(增)'))
        for s in seqs:
            a = r_aug.get(s, {})
            n = r_noaug.get(s, {})
            f.write('%-15s %-10.4f %-10.4f %-10.4f %-10.4f %-10.1f\n' %
                    (s, a.get('iou', 0), n.get('iou', 0),
                     a.get('prec', 0), n.get('prec', 0), 0))
        if r_aug and r_noaug:
            f.write('\n平均:')
            for k in ['iou', 'success', 'prec']:
                av = np.mean([r.get(k, 0) for r in r_aug.values()])
                nv = np.mean([r.get(k, 0) for r in r_noaug.values()]) if r_noaug else 0
                f.write('  %s: 增强=%.4f 无增强=%.4f' % (k, av, nv))

    # 画图
    names = [s for s in seqs if s in r_aug]
    if not names: return
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(names))
    w = 0.35
    aug_vals = [r_aug[s]['iou'] for s in names]
    noaug_vals = [r_noaug.get(s, {}).get('iou', 0) for s in names]
    ax.bar(x - w/2, aug_vals, w, label='有增强', color='steelblue')
    ax.bar(x + w/2, noaug_vals, w, label='无增强', color='coral')
    ax.set_ylabel('平均IoU')
    ax.set_title('数据增强对比实验 (epcoch 30)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print('对比结果已保存: %s' % OUTPUT_FILE)


if __name__ == '__main__':
    main()