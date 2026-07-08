# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import IoU, corner2center
from pysot.utils.anchor import Anchors


class AnchorTarget:
    def __init__(self,):
        self.anchors = Anchors(cfg.ANCHOR.STRIDE,
                               cfg.ANCHOR.RATIOS,
                               cfg.ANCHOR.SCALES)

        self.anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)

    def __call__(self, target, size, neg=False):
        anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)

        # 获取 anchors 实际生成的网格尺寸
        anchor_box = self.anchors.all_anchors[0]
        actual_size = anchor_box.shape[-1]

        # 使用实际尺寸创建数组，避免尺寸不匹配
        cls = -1 * np.ones((anchor_num, actual_size, actual_size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, actual_size, actual_size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, actual_size, actual_size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        if neg:
            cx = actual_size // 2
            cy = actual_size // 2
            cx += int(np.ceil((tcx - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            cy += int(np.ceil((tcy - cfg.TRAIN.SEARCH_SIZE // 2) /
                      cfg.ANCHOR.STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(actual_size, cx + 4)
            u = max(0, cy - 3)
            d = min(actual_size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, neg_num = select(np.where(cls == 0), cfg.TRAIN.NEG_NUM)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, actual_size, actual_size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_center = self.anchors.all_anchors[1]
        cx, cy, w, h = anchor_center[0], anchor_center[1], anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], anchor_box[2], anchor_box[3]
        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > cfg.TRAIN.THR_HIGH)
        neg = np.where(overlap < cfg.TRAIN.THR_LOW)

        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap