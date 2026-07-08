"""SiamRPN++封装"""
import sys, os
sys.path.insert(0, r'D:\SiamFC_RPN\SiamRPNpp')
import cv2
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain

SIAMRPNPP_CONFIG = r'D:\SiamFC_RPN\SiamRPNpp\experiments\siamrpn_r50_l234_dwxcorr\config.yaml'
SIAMRPNPP_MODEL = r'D:\SiamFC_RPN\SiamRPNpp\experiments\siamrpn_r50_l234_dwxcorr\model.pth'

class SiamRPNppWrapper:
    def __init__(self):
        cfg.merge_from_file(SIAMRPNPP_CONFIG)
        self.model = ModelBuilder()
        self.model = load_pretrain(self.model, SIAMRPNPP_MODEL).cuda().eval()
        self.tracker = build_tracker(self.model)
        self.initialized = False

    def init(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        cx, cy, w2, h2 = get_axis_aligned_bbox(np.array([x, y, w, h]))
        init_box = [cx-(w2-1)/2, cy-(h2-1)/2, w2, h2]
        self.tracker.init(frame, init_box)
        self.initialized = True
        return [x, y, w, h]

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError('Tracker not initialized')
        outputs = self.tracker.track(frame)
        return [float(v) for v in outputs['bbox']]

    def get_name(self):
        return 'SiamRPN++'