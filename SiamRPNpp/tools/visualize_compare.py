"""
SiamFC vs SiamRPN++ 对比可视化脚本
生成课设需要的所有图表
"""
import os, sys, cv2, numpy as np, glob, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===== 路径配置 =====
TEST_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
SIAMFC_RESULT_DIR = r'D:\SiamFC_RPN\SiamFC\tools\results\OTB2015\SiamFC'
SIAMRPNPP_RESULT_DIR = r'D:\SiamFC_RPN\SiamRPNpp\results\OTB100\SiamRPNpp'
OUTPUT_DIR = r'D:\SiamFC_RPN\SiamRPNpp\results\visualizations'

# 20个测试视频
SELECTED_SEQS = [
    'Basketball', 'Biker', 'BlurCar2', 'Jumping',
    'Coke', 'Girl', 'Jogging', 'Tiger2',
    'CarScale', 'David', 'Dog', 'Singer1',
    'Bolt', 'Dancer', 'Freeman1', 'Skater',
    'Board', 'Deer', 'Liquor', 'Soccer'
]

# 难度分类（用于失败案例选择）
DIFFICULTY = {
    'fast_motion': ['Basketball', 'Biker', 'BlurCar2', 'Jumping'],
    'occlusion': ['Coke', 'Girl', 'Jogging', 'Tiger2'],
    'scale_change': ['CarScale', 'David', 'Dog', 'Singer1'],
    'deformation': ['Bolt', 'Dancer', 'Freeman1', 'Skater'],
    'background_clutter': ['Board', 'Deer', 'Liquor', 'Soccer']
}

# SiamFC 测试结果（从test_results.txt提取）
SIAMFC_RESULTS = {
    'Basketball': (0.5118, 0.5760, 0.6188, 26.23),
    'Biker': (0.6526, 0.8582, 0.9716, 3.70),
    'BlurCar2': (0.8563, 1.0000, 1.0000, 4.25),
    'Jumping': (0.4767, 0.6795, 0.6859, 20.04),
    'Coke': (0.4348, 0.3655, 0.6724, 25.59),
    'Girl': (0.6840, 0.9238, 1.0000, 2.92),
    'Jogging': (0.6671, 0.9608, 0.9608, 12.01),
    'Tiger2': (0.6005, 0.7473, 0.7500, 16.29),
    'CarScale': (0.6830, 0.6932, 0.6853, 19.16),
    'David': (0.0760, 0.0383, 0.0681, 57.33),
    'Dog': (0.3651, 0.0952, 0.9841, 6.87),
    'Singer1': (0.7382, 1.0000, 0.9943, 7.10),
    'Bolt': (0.3843, 0.3983, 0.6734, 68.99),
    'Dancer': (0.7820, 1.0000, 0.9955, 7.77),
    'Freeman1': (0.5260, 0.5046, 0.9231, 8.42),
    'Skater': (0.6562, 0.9371, 1.0000, 7.48),
    'Board': (0.1111, 0.1263, 0.0660, 252.65),
    'Deer': (0.7753, 1.0000, 1.0000, 4.76),
    'Liquor': (0.3399, 0.4000, 0.4023, 93.84),
    'Soccer': (0.1427, 0.1407, 0.1739, 78.05),
}


def read_tracking_result(filepath):
    """读取跟踪结果文件 (x1,y1,w,h per line) - 逗号分隔"""
    data = np.loadtxt(filepath, delimiter=',')
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    return data

def read_gt_file(filepath):
    """读取groundtruth文件，兼容逗号和制表符"""
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except:
        try:
            data = np.loadtxt(filepath, delimiter='\t')
        except:
            data = np.loadtxt(filepath)
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    return data


def compute_iou(pred_box, gt_box):
    x1 = max(pred_box[0], gt_box[0])
    y1 = max(pred_box[1], gt_box[1])
    x2 = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    y2 = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = pred_box[2] * pred_box[3]
    area2 = gt_box[2] * gt_box[3]
    union = area1 + area2 - inter
    return inter / max(union, 1e-16)


def plot_frame_comparison(ax, img, gt_box, pred1_box, pred2_box, label1, label2, title):
    """在一张图上绘制GT+两个模型的预测框"""
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w = img.shape[:2]
    # GT框（红色）
    rect_gt = Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3],
                         linewidth=2, edgecolor='red', facecolor='none', label='GT')
    ax.add_patch(rect_gt)
    # 模型1（绿色）
    if pred1_box is not None:
        rect_p1 = Rectangle((pred1_box[0], pred1_box[1]), pred1_box[2], pred1_box[3],
                             linewidth=2, edgecolor='lime', facecolor='none', label=label1)
        ax.add_patch(rect_p1)
    # 模型2（黄色）
    if pred2_box is not None:
        rect_p2 = Rectangle((pred2_box[0], pred2_box[1]), pred2_box[2], pred2_box[3],
                             linewidth=2, edgecolor='yellow', facecolor='none', label=label2)
        ax.add_patch(rect_p2)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=6)


def generate_comparison_images():
    """生成5组成功案例对比图+3组失败案例"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'success_cases'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'failure_cases'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'comparison_bars'), exist_ok=True)

    # 计算每个视频的平均IoU
    seq_ious = {}
    for seq_name in SELECTED_SEQS:
        result_file = os.path.join(SIAMRPNPP_RESULT_DIR, f'{seq_name.lower()}.txt')
        if not os.path.exists(result_file):
            continue
        pred = read_tracking_result(result_file)
        anno_file = os.path.join(TEST_ROOT, seq_name, 'groundtruth_rect.txt')
        gt = read_gt_file(anno_file)
        if len(gt.shape) == 1:
            gt = gt.reshape(1, -1)
        n = min(len(pred), len(gt))
        ious = [compute_iou(pred[t], gt[t]) for t in range(1, n)]
        seq_ious[seq_name] = (np.mean(ious), seq_name)

    # 按IoU排序选5个最好和3个最差
    sorted_seqs = sorted(seq_ious.items(), key=lambda x: x[1][0], reverse=True)
    best_5 = sorted_seqs[:5]
    worst_3 = sorted_seqs[-3:]

    print('5个最佳序列:')
    for seq, (iou, _) in best_5:
        print(f'  {seq}: IoU={iou:.4f}')
    print('3个最差序列:')
    for seq, (iou, _) in worst_3:
        print(f'  {seq}: IoU={iou:.4f}')

    # 生成5个成功案例
    for idx, (seq_name, (avg_iou, _)) in enumerate(best_5):
        _generate_case_folder(seq_name, idx+1, avg_iou, 'success_cases', '成功')

    # 生成3个失败案例
    for idx, (seq_name, (avg_iou, _)) in enumerate(worst_3):
        category = ''
        for cat, seqs in DIFFICULTY.items():
            if seq_name in seqs:
                category = cat
                break
        _generate_case_folder(seq_name, idx+1, avg_iou, 'failure_cases', '失败', category)

    # 生成对比柱状图
    generate_bar_chart(sorted_seqs)


def _generate_case_folder(seq_name, case_idx, avg_iou, case_type, cn_label, category=''):
    """生成单个案例的所有图表"""
    case_dir = os.path.join(OUTPUT_DIR, case_type, f'case{case_idx}_{seq_name}')
    os.makedirs(case_dir, exist_ok=True)

    # 读取数据
    img_dir = os.path.join(TEST_ROOT, seq_name, 'img')
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    anno_file = os.path.join(TEST_ROOT, seq_name, 'groundtruth_rect.txt')
    gt = read_gt_file(anno_file)
    if len(gt.shape) == 1:
        gt = gt.reshape(1, -1)

    fc_result_file = os.path.join(SIAMFC_RESULT_DIR, f'{seq_name.lower()}.txt')
    rpn_result_file = os.path.join(SIAMRPNPP_RESULT_DIR, f'{seq_name.lower()}.txt')
    fc_pred = read_tracking_result(fc_result_file) if os.path.exists(fc_result_file) else None
    rpn_pred = read_tracking_result(rpn_result_file) if os.path.exists(rpn_result_file) else None

    n = min(len(img_files), len(gt))
    if fc_pred is not None:
        n = min(n, len(fc_pred))
    if rpn_pred is not None:
        n = min(n, len(rpn_pred))

    # 计算逐帧IoU
    fc_ious = [compute_iou(fc_pred[t], gt[t]) for t in range(1, n)] if fc_pred is not None else []
    rpn_ious = [compute_iou(rpn_pred[t], gt[t]) for t in range(1, n)] if rpn_pred is not None else []
    fc_cles = [np.sqrt((fc_pred[t][0]+fc_pred[t][2]/2-gt[t][0]-gt[t][2]/2)**2 +
                        (fc_pred[t][1]+fc_pred[t][3]/2-gt[t][1]-gt[t][3]/2)**2)
               for t in range(1, n)] if fc_pred is not None else []
    rpn_cles = [np.sqrt((rpn_pred[t][0]+rpn_pred[t][2]/2-gt[t][0]-gt[t][2]/2)**2 +
                         (rpn_pred[t][1]+rpn_pred[t][3]/2-gt[t][1]-gt[t][3]/2)**2)
                for t in range(1, n)] if rpn_pred is not None else []

    # 1. 首帧+中间帧局部放大对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    frame_indices = [0, min(n-1, n//2), min(n-1, 100)]
    titles = ['首帧', '中间帧', '典型帧']
    for ax, fid, t in zip(axes, frame_indices, titles):
        img = cv2.imread(img_files[fid])
        if img is None:
            continue
        fc_b = fc_pred[fid] if fc_pred is not None and fid < len(fc_pred) else None
        rpn_b = rpn_pred[fid] if rpn_pred is not None and fid < len(rpn_pred) else None
        plot_frame_comparison(ax, img, gt[fid], fc_b, rpn_b, 'SiamFC', 'SiamRPN++', f'{t} (帧{fid})')
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, 'frame_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. IoU曲线
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, n)
    if fc_ious:
        ax.plot(x, fc_ious, 'g-', alpha=0.7, label=f'SiamFC (Avg={np.mean(fc_ious):.4f})', linewidth=1)
    if rpn_ious:
        ax.plot(x, rpn_ious, 'b-', alpha=0.7, label=f'SiamRPN++ (Avg={np.mean(rpn_ious):.4f})', linewidth=1)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Success阈值(0.5)')
    ax.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='失败阈值(0.3)')
    ax.set_xlabel('帧号')
    ax.set_ylabel('IoU')
    ax.set_title(f'{seq_name} - IoU曲线 ({cn_label}案例)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, 'iou_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. CLE曲线
    fig, ax = plt.subplots(figsize=(10, 4))
    if fc_cles:
        ax.plot(x, fc_cles, 'g-', alpha=0.7, label=f'SiamFC (Avg={np.mean(fc_cles):.1f}px)', linewidth=1)
    if rpn_cles:
        ax.plot(x, rpn_cles, 'b-', alpha=0.7, label=f'SiamRPN++ (Avg={np.mean(rpn_cles):.1f}px)', linewidth=1)
    ax.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Precision阈值(20px)')
    ax.set_xlabel('帧号')
    ax.set_ylabel('中心位置误差 (像素)')
    ax.set_title(f'{seq_name} - CLE曲线 ({cn_label}案例)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, 'cle_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Success Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    thresholds = np.linspace(0, 1, 100)
    if fc_ious:
        fc_success = [np.mean(np.array(fc_ious) > t) for t in thresholds]
        ax.plot(thresholds, fc_success, 'g-', linewidth=2, label=f'SiamFC (AUC={np.trapz(fc_success, thresholds):.4f})')
    if rpn_ious:
        rpn_success = [np.mean(np.array(rpn_ious) > t) for t in thresholds]
        ax.plot(thresholds, rpn_success, 'b-', linewidth=2, label=f'SiamRPN++ (AUC={np.trapz(rpn_success, thresholds):.4f})')
    ax.set_xlabel('重叠阈值')
    ax.set_ylabel('成功率')
    ax.set_title(f'{seq_name} - Success Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, 'success_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. 文本报告
    fc_iou = np.mean(fc_ious) if fc_ious else 0
    fc_cle = np.mean(fc_cles) if fc_cles else 0
    fc_success = np.mean(np.array(fc_ious) > 0.5) if fc_ious else 0
    rpn_iou = np.mean(rpn_ious) if rpn_ious else 0
    rpn_cle = np.mean(rpn_cles) if rpn_cles else 0
    rpn_success = np.mean(np.array(rpn_ious) > 0.5) if rpn_ious else 0
    fc_fail = np.mean(np.array(fc_ious) < 0.3) if fc_ious else 0
    rpn_fail = np.mean(np.array(rpn_ious) < 0.3) if rpn_ious else 0

    with open(os.path.join(case_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{"="*60}\n')
        f.write(f'{seq_name} - {cn_label}案例 #{case_idx}\n')
        f.write(f'{"="*60}\n\n')
        f.write(f'序列信息:\n')
        f.write(f'  总帧数: {n}\n')
        f.write(f'  难度类别: {category if category else "通用"}\n')
        f.write(f'  平均IoU (SiamRPN++): {rpn_iou:.4f}\n\n')
        f.write(f'指标对比:\n')
        f.write(f'  {"指标":<20s} {"SiamFC":<12s} {"SiamRPN++":<12s}\n')
        f.write(f'  {"-"*44}\n')
        f.write(f'  {"平均IoU":<20s} {fc_iou:<12.4f} {rpn_iou:<12.4f}\n')
        f.write(f'  {"成功率(IoU>0.5)":<20s} {fc_success:<12.4f} {rpn_success:<12.4f}\n')
        f.write(f'  {"平均CLE(px)":<20s} {fc_cle:<12.1f} {rpn_cle:<12.1f}\n')
        f.write(f'  {"失败率(IoU<0.3)":<20s} {fc_fail:<12.4f} {rpn_fail:<12.4f}\n\n')
        if case_type == 'failure_cases':
            f.write(f'失败分析:\n')
            f.write(f'  失败帧占比: {rpn_fail:.1%}\n')
            f.write(f'  主要原因: {category}\n')

    print(f'  [生成] {case_type}/{case_idx}_{seq_name}')


def generate_bar_chart(sorted_seqs):
    """生成SiamFC vs SiamRPN++ 对比柱状图"""
    seq_names = [s[0] for s in sorted_seqs]
    rpn_ious = [s[1][0] for s in sorted_seqs]
    fc_ious = [SIAMFC_RESULTS[s[0]][0] for s in sorted_seqs]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(seq_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, fc_ious, width, label='SiamFC', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, rpn_ious, width, label='SiamRPN++', color='blue', alpha=0.7)
    ax.set_xlabel('测试序列')
    ax.set_ylabel('IoU')
    ax.set_title('SiamFC vs SiamRPN++ 各序列IoU对比')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    # 在柱子上标数值
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=7, rotation=45)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=7, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_bars', 'iou_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 生成20个序列的指标表
    print('\n生成综合指标表...')
    avg_fc_iou = np.mean(fc_ious)
    avg_rpn_iou = np.mean(rpn_ious)
    avg_fc_success = np.mean([SIAMFC_RESULTS[s[0]][1] for s in sorted_seqs])
    avg_rpn_success = np.mean([SIAMFC_RESULTS[s[0]][1] if s[0] in SIAMFC_RESULTS else 0 for s in sorted_seqs])
    avg_fc_prec = np.mean([SIAMFC_RESULTS[s[0]][2] for s in sorted_seqs])
    avg_rpn_prec = np.mean([SIAMFC_RESULTS[s[0]][2] if s[0] in SIAMFC_RESULTS else 0 for s in sorted_seqs])

    with open(os.path.join(OUTPUT_DIR, 'comparison_table.txt'), 'w', encoding='utf-8') as f:
        f.write('SiamFC vs SiamRPN++ 对比实验综合表\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'{"序列":<15s} {"SiamFC IoU":<12s} {"RPN++ IoU":<12s} {"SiamFC Succ":<12s} {"RPN++ Succ":<12s}\n')
        f.write('-' * 63 + '\n')
        for seq_name in SELECTED_SEQS:
            if seq_name in SIAMFC_RESULTS:
                fc_iou = SIAMFC_RESULTS[seq_name][0]
                fc_succ = SIAMFC_RESULTS[seq_name][1]
            else:
                fc_iou, fc_succ = 0, 0
            # 从结果文件计算rpn指标
            result_file = os.path.join(SIAMRPNPP_RESULT_DIR, f'{seq_name.lower()}.txt')
            if os.path.exists(result_file):
                pred = read_tracking_result(result_file)
                anno_file = os.path.join(TEST_ROOT, seq_name, 'groundtruth_rect.txt')
                gt = read_gt_file(anno_file)
                if len(gt.shape) == 1:
                    gt = gt.reshape(1, -1)
                n = min(len(pred), len(gt))
                ious = [compute_iou(pred[t], gt[t]) for t in range(1, n)]
                rpn_iou_v = np.mean(ious)
                rpn_succ_v = np.mean(np.array(ious) > 0.5)
            else:
                rpn_iou_v, rpn_succ_v = 0, 0
            f.write(f'{seq_name:<15s} {fc_iou:<12.4f} {rpn_iou_v:<12.4f} {fc_succ:<12.4f} {rpn_succ_v:<12.4f}\n')

        f.write('\n' + '=' * 70 + '\n')
        f.write(f'总体平均对比:\n')
        f.write(f'  IoU:     SiamFC={avg_fc_iou:.4f}  SiamRPN++={avg_rpn_iou:.4f}  提升={((avg_rpn_iou-avg_fc_iou)/max(avg_fc_iou,0.001))*100:.1f}%\n')
        f.write(f'  Success: SiamFC={avg_fc_success:.4f}  SiamRPN++={avg_rpn_success:.4f}\n')
        f.write(f'  Precision: SiamFC={avg_fc_prec:.4f}  SiamRPN++={avg_rpn_prec:.4f}\n')

    print(f'  对比表已保存')


def generate_failure_analysis():
    """生成3个失败案例的详细分析"""
    worst_3 = ['Board', 'David', 'Soccer']  # 从之前结果选3个最差的
    reasons = {
        'Board': ('背景干扰', '滑板场景中背景与目标极度相似，特征难以区分'),
        'David': ('跟踪漂移', 'SiamFC无法处理光照变化和表情变化，导致目标丢失'),
        'Soccer': ('背景干扰', '足球比赛中运动员外观相似，简单特征无法区分')
    }

    for seq_name in worst_3:
        img_dir = os.path.join(TEST_ROOT, seq_name, 'img')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if not img_files:
            continue

        # 找丢失帧：IoU最低的帧
        fc_result = os.path.join(SIAMFC_RESULT_DIR, f'{seq_name.lower()}.txt')
        rpn_result = os.path.join(SIAMRPNPP_RESULT_DIR, f'{seq_name.lower()}.txt')
        anno_file = os.path.join(TEST_ROOT, seq_name, 'groundtruth_rect.txt')
        gt = read_gt_file(anno_file)

        if os.path.exists(fc_result):
            fc_pred = read_tracking_result(fc_result)
            fc_ious = [compute_iou(fc_pred[t], gt[t]) for t in range(1, min(len(fc_pred), len(gt)))]
            worst_fc_frame = np.argmin(fc_ious) + 1 if fc_ious else 0
        else:
            worst_fc_frame = 0

        if worst_fc_frame < len(img_files):
            img = cv2.imread(img_files[worst_fc_frame])
            fc_b = fc_pred[worst_fc_frame] if os.path.exists(fc_result) else None
            rpn_b = read_tracking_result(rpn_result)[worst_fc_frame] if os.path.exists(rpn_result) else None

            fig, ax = plt.subplots(figsize=(10, 8))
            plot_frame_comparison(ax, img, gt[worst_fc_frame], fc_b, rpn_b, 'SiamFC', 'SiamRPN++',
                                   f'{seq_name} - 丢失帧 #{worst_fc_frame}')
            plt.tight_layout()
            out_dir = os.path.join(OUTPUT_DIR, 'failure_cases', f'failure_{seq_name}')
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, 'failure_frame.png'), dpi=150, bbox_inches='tight')
            plt.close()

    print('  失败案例帧已生成')


if __name__ == '__main__':
    print('开始生成SiamFC vs SiamRPN++ 可视化对比...')
    generate_comparison_images()
    generate_failure_analysis()
    print(f'\n所有可视化结果已保存至: {OUTPUT_DIR}')