"""SiamFC封装"""
import sys, os
sys.path.insert(0, r'D:\SiamFC_RPN\SiamFC')
import cv2
import numpy as np
from siamfc.siamfc import TrackerSiamFC

SIAMFC_MODEL_PATH = r'D:\SiamFC_RPN\SiamFC\snapshot_mytrain\siamfc_alexnet_e30.pth'

class SiamFCWrapper:
    def __init__(self):
        self.tracker = TrackerSiamFC(net_path=SIAMFC_MODEL_PATH)
        self.initialized = False

    def init(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        self.tracker.init(frame, (x, y, w, h))
        self.initialized = True
        return [x, y, w, h]

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError('Tracker not initialized')
        box = self.tracker.update(frame)
        return [float(v) for v in box]

    def get_name(self):
        return 'SiamFC'