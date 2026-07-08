import cv2
import numpy as np
import time
from models.base_tracker import BaseTracker


class EdgeContourTracker(BaseTracker):
    """
    纯边缘轮廓法跟踪器。
    Canny边缘检测 → 膨胀连接 → 轮廓提取 → 多条件评分筛选最佳目标框。
    不使用任何OpenCV内置跟踪器（如CamShift）。
    """
    def __init__(self, tracker_name="edge_contour"):
        super().__init__(tracker_name)

        self.init_bbox = None
        self.prev_bbox = None
        self.target_area = 0
        self.target_aspect = 1.0
        self.lost_count = 0
        self.frame_count = 0
        self.start_time = None

        # Canny参数（自适应）
        self.canny_low = 50
        self.canny_high = 150

        # 形态学核
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def init(self, frame, bbox):
        self.init_bbox = bbox
        self.prev_bbox = bbox

        x, y, w, h = [int(v) for v in bbox]
        self.target_area = w * h
        self.target_aspect = w / (h + 1e-6)

        # 自适应Canny阈值
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        median = np.median(roi_gray)
        self.canny_low = max(10, int(median * 0.3))
        self.canny_high = min(255, int(median * 1.0))
        if self.canny_low >= self.canny_high:
            self.canny_low = max(5, self.canny_high - 15)

        self.start_time = time.time()
        self.frame_count = 1
        self.trajectory = [bbox]

    def _center(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2])

    def _extract_edges(self, gray):
        """多尺度Canny边缘检测 + 形态学处理"""
        # 用两组阈值检测边缘，然后合并
        edges_low = cv2.Canny(gray, max(5, self.canny_low // 2), self.canny_high)
        edges_mid = cv2.Canny(gray, self.canny_low, self.canny_high)
        edges_high = cv2.Canny(gray, min(255, self.canny_low * 2), min(255, self.canny_high + 30))

        edges = cv2.bitwise_or(edges_low, edges_mid)
        edges = cv2.bitwise_or(edges, edges_high)

        # 膨胀连接断裂边缘
        edges = cv2.dilate(edges, self.dilate_kernel, iterations=2)
        # 闭运算填充小空洞
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.close_kernel, iterations=1)

        return edges

    def _find_best_contour(self, edges, prev_bbox, frame_shape):
        """在边缘图中找最佳目标轮廓"""
        x, y, w, h = prev_bbox
        max_dim = max(w, h)

        # 多尺度搜索
        margins = [0.5, 1.0, 1.5, 2.5, 4.0]
        if self.lost_count > 3:
            margins = [1.0, 2.0, 4.0, 8.0]

        # 全图轮廓
        all_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not all_contours:
            return None

        prev_center = self._center(prev_bbox)
        best_box = None
        best_score = 0

        for margin in margins:
            x1 = max(0, x - int(max_dim * margin))
            y1 = max(0, y - int(max_dim * margin))
            x2 = min(frame_shape[1], x + w + int(max_dim * margin))
            y2 = min(frame_shape[0], y + h + int(max_dim * margin))

            for c in all_contours:
                area = cv2.contourArea(c)
                # 边缘轮廓面积通常远小于目标面积，用很宽松的下限
                if area < max(30, self.target_area * 0.003) or area > self.target_area * 8:
                    continue

                bx, by, bw, bh = cv2.boundingRect(c)
                # 检查是否在当前搜索范围内
                if not (x1 <= bx + bw//2 <= x2 and y1 <= by + bh//2 <= y2):
                    continue

                aspect = bw / (bh + 1e-6)
                if abs(aspect - self.target_aspect) > 5:
                    continue

                center = np.array([bx + bw/2, by + bh/2])
                dist = np.linalg.norm(center - prev_center)

                # 评分
                dist_score = 1.0 / (1.0 + dist / max_dim)
                area_score = min(area / self.target_area, self.target_area / area)
                aspect_score = 1.0 / (1.0 + abs(aspect - self.target_aspect))

                score = 0.5 * dist_score + 0.3 * area_score + 0.2 * aspect_score

                if score > best_score:
                    best_score = score
                    best_box = (int(bx), int(by), int(bw), int(bh))

            if best_box is not None:
                break  # 找到就返回，不再扩大搜索

        return best_box

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 动态更新Canny阈值（每帧基于prev_bbox区域）
        x, y, w, h = self.prev_bbox
        if w > 5 and h > 5:
            roi_gray = gray[max(0,y):min(frame.shape[0],y+h), max(0,x):min(frame.shape[1],x+w)]
            if roi_gray.size > 0:
                median = np.median(roi_gray)
                self.canny_low = max(10, int(median * 0.3))
                self.canny_high = min(255, int(median * 1.0))
                if self.canny_low >= self.canny_high:
                    self.canny_low = max(5, self.canny_high - 15)

        # 边缘检测
        edges = self._extract_edges(gray)

        # 找最佳轮廓
        cand = self._find_best_contour(edges, self.prev_bbox, frame.shape)

        if cand is None:
            cand = self.prev_bbox

        # 有效性验证
        c1 = self._center(self.prev_bbox)
        c2 = self._center(cand)
        dist = np.linalg.norm(c1 - c2)
        max_dim = max(self.prev_bbox[2], self.prev_bbox[3])
        new_area = cand[2] * cand[3]

        if dist < max_dim * 5 and 0.02 < new_area / (self.target_area + 1e-6) < 10:
            self.prev_bbox = cand
            self.lost_count = 0
        else:
            self.lost_count += 1

        self.frame_count += 1
        self.trajectory.append(self.prev_bbox)
        return self.prev_bbox

    def get_metrics(self):
        fps = self.frame_count / (time.time() - self.start_time + 1e-6)
        return {"fps": fps, "lost": self.lost_count}