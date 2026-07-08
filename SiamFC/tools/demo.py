from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    # 修改为你OTB100中的某个序列路径，例如Crossing或其他序列
    seq_dir = r'D:\jupyter\tracking\OTB100\Crossing'
    img_files = sorted(glob.glob(seq_dir + '/img/*.jpg'))
    anno = np.loadtxt(seq_dir + '/groundtruth_rect.txt')

    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
