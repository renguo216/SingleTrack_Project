"""从OTB图片序列合成视频，保留groundtruth文件"""
import os, cv2, glob, shutil

TEST_ROOT = r'D:\experiment\SingleTrack_Project\data\datasets\test'
OUTPUT_DIR = r'D:\SiamFC_RPN\tracking_web\otb_videos'

# 选30个具有代表性的序列（覆盖不同难度）
SELECTED_30 = [
    'Basketball', 'BlurCar2', 'Jumping', 'Biker',          # 快速运动
    'Coke', 'Girl', 'Jogging', 'Tiger2',                     # 遮挡
    'CarScale', 'David', 'Dog', 'Singer1',                    # 尺度变化
    'Bolt', 'Dancer', 'Freeman1', 'Skater',                   # 形变/旋转
    'Board', 'Deer', 'Liquor', 'Soccer',                      # 背景干扰
    'Bird2', 'BlurBody', 'Box', 'CarDark',                    # 补充
    'Couple', 'DragonBaby', 'FaceOcc1', 'Football',
    'Human3', 'Lemming'
]

def make_video(seq_name):
    seq_dir = os.path.join(TEST_ROOT, seq_name)
    img_dir = os.path.join(seq_dir, 'img')
    if not os.path.isdir(img_dir):
        return None
    
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not img_files:
        return None
    
    gt_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
    if not os.path.exists(gt_file):
        return None
    
    # 读取第一帧确定尺寸
    first = cv2.imread(img_files[0])
    h, w = first.shape[:2]
    
    # 输出路径
    out_dir = os.path.join(OUTPUT_DIR, seq_name)
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, f'{seq_name}.mp4')
    
    # 视频写入器（15fps，原始尺寸）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 15, (w, h))
    
    for img_file in img_files:
        frame = cv2.imread(img_file)
        if frame is not None:
            writer.write(frame)
    
    writer.release()
    
    # 复制groundtruth
    shutil.copy2(gt_file, os.path.join(out_dir, 'groundtruth_rect.txt'))
    
    n_frames = len(img_files)
    print(f'  {seq_name}: {n_frames}帧, {w}x{h}, {video_path}')
    return n_frames


if __name__ == '__main__':
    print(f'合成 {len(SELECTED_30)} 个OTB视频到 {OUTPUT_DIR}')
    print('=' * 60)
    total = 0
    for seq in SELECTED_30:
        n = make_video(seq)
        if n:
            total += n
    print('=' * 60)
    print(f'完成! 共 {total} 帧')