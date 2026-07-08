from __future__ import absolute_import

import os, glob, numpy as np, cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from siamfc import TrackerSiamFC

net_path = 'snapshot_mytrain/siamfc_alexnet_e50.pth'
output_dir = 'results_videos'
dataset_root = r'D:\experiment\SingleTrack_Project\data\datasets\test'

tracker = TrackerSiamFC(net_path=net_path)

# Only failed sequences
failed = ['Shaking', 'Singer1', 'Singer2', 'Skater', 'Skater2', 'Skating1', 
          'Skiing', 'Soccer', 'Subway', 'Surfer', 'Suv', 'Sylvester',
          'Tiger1', 'Tiger2', 'Toy', 'Trans', 'Trellis', 'Twinnings', 
          'Vase', 'Walking', 'Walking2', 'Woman', 'BlurCar1']

for seq_name in failed:
    seq_dir = os.path.join(dataset_root, seq_name)
    if not os.path.isdir(seq_dir):
        print('Skip {}: dir not found'.format(seq_name))
        continue
    
    img_files = sorted(glob.glob(os.path.join(seq_dir, 'img', '*.jpg')))
    if len(img_files) == 0:
        img_files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    if len(img_files) == 0:
        print('Skip {}: no images'.format(seq_name))
        continue
    
    anno_file = os.path.join(seq_dir, 'groundtruth_rect.txt')
    if not os.path.isfile(anno_file):
        print('Skip {}: no annotation'.format(seq_name))
        continue
    
    try:
        try:
            anno = np.loadtxt(anno_file, delimiter=',')
        except:
            anno = np.loadtxt(anno_file)
    except Exception as e:
        print('Skip {}: anno error {}'.format(seq_name, e))
        continue
    
    print('Processing {} ({} frames)...'.format(seq_name, len(img_files)))
    try:
        boxes, times = tracker.track(img_files, anno[0], visualize=False)
    except Exception as e:
        print('FAILED {}: {}'.format(seq_name, e))
        continue
    
    video_path = os.path.join(output_dir, seq_name + '.avi')
    first_img = cv2.imread(img_files[0])
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
    
    n = min(len(img_files), len(anno))
    for i in range(n):
        img = cv2.imread(img_files[i])
        if img is None: continue
        gt = anno[i]
        cv2.rectangle(img, (int(gt[0]),int(gt[1])), (int(gt[0]+gt[2]),int(gt[1]+gt[3])), (0,0,255), 2)
        tr = boxes[i]
        cv2.rectangle(img, (int(tr[0]),int(tr[1])), (int(tr[0]+tr[2]),int(tr[1]+tr[3])), (0,255,0), 2)
        cv2.putText(img, seq_name, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        vw.write(img)
    vw.release()
    print('OK {} saved. FPS: {:.1f}'.format(seq_name, len(img_files)/sum(times)))