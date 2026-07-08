"""传统方法封装 - 直接调用 traditional/src 下的实现"""
import sys, os
import cv2
import numpy as np

# 添加传统方法路径并直接导入class（绕过__init__.py的models路径问题）
TRAD_DIR = r'D:\SiamFC_RPN\traditional\src'
sys.path.insert(0, TRAD_DIR)

# 直接导入 tracker 类
from trackers.frame_diff_tracker import FrameDiffTracker
from trackers.background_subtraction_tracker import BackgroundSubtractionTracker
from trackers.color_threshold_tracker import ColorThresholdTracker
from trackers.edge_contour_tracker import EdgeContourTracker


TRADITIONAL_METHODS = {
    'frame_diff': {'name': '帧差法', 'class': FrameDiffTracker},
    'background_sub': {'name': '背景差减法', 'class': BackgroundSubtractionTracker},
    'color_threshold': {'name': '颜色阈值法', 'class': ColorThresholdTracker},
    'edge_contour': {'name': '边缘轮廓跟踪法', 'class': EdgeContourTracker},
}


class TraditionalTrackerWrapper:
    def __init__(self, method_name):
        if method_name not in TRADITIONAL_METHODS:
            raise ValueError(f'未知方法: {method_name}')
        self.info = TRADITIONAL_METHODS[method_name]
        self.tracker = self.info['class']()
        self.initialized = False

    def init(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        self.tracker.init(frame, (x, y, w, h))
        self.initialized = True
        return [x, y, w, h]

    def update(self, frame):
        if not self.initialized:
            raise RuntimeError('Tracker not initialized')
        bbox = self.tracker.update(frame)
        return [int(v) for v in bbox]

    def get_name(self):
        return self.info['name']