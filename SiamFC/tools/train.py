from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    root_dir = r'D:\experiment\SingleTrack_Project\data\datasets'
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    # 使用优化后的训练参数
    tracker = TrackerSiamFC(
        epoch_num=50,
        batch_size=8,
        initial_lr=1e-2,
        ultimate_lr=1e-5,
        weight_decay=5e-4,
        momentum=0.9,
        num_workers=8  # 根据你的CPU核心数调整，Windows建议不要太大
    )
    tracker.train_over(seqs, save_dir='snapshot_mytrain')
