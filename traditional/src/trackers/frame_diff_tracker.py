import cv2
import numpy as np
from models.base_tracker import BaseTracker


class FrameDiffTracker(BaseTracker):
    def __init__(self):
        super().__init__("frame_diff")

        self.prev_frame = None
        self.init_bbox = None
        self.prev_bbox = None
        self.target_area = 0
        self.target_aspect = 1.0
        self.lost_count = 0

    def init(self, frame, bbox):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        self.prev_frame = gray
        self.init_bbox = bbox
        self.prev_bbox = bbox
        self.target_area = bbox[2] * bbox[3]
        self.target_aspect = bbox[2] / (bbox[3] + 1e-6)
        self.trajectory = [bbox]

    def _center(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2])

    def _motion_bbox(self, diff, frame_shape):
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1
        prev_center = self._center(self.prev_bbox)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            if area < self.target_area * 0.1 or area > self.target_area * 5:
                continue

            aspect = w / (h + 1e-6)
            if abs(aspect - self.target_aspect) > 2:
                continue

            center = self._center((x, y, w, h))
            dist = np.linalg.norm(center - prev_center)
            
            max_dim = max(self.prev_bbox[2], self.prev_bbox[3])
            if dist > max_dim * 2:
                continue

            distance_score = 1.0 / (1.0 + dist / max_dim)
            area_score = min(area / self.target_area, self.target_area / area)
            aspect_score = 1.0 / (1.0 + abs(aspect - self.target_aspect))
            
            score = 0.5 * distance_score + 0.3 * area_score + 0.2 * aspect_score

            if score > best_score:
                best_score = score
                best = (x, y, w, h)

        return best

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(self.prev_frame, gray)
        
        _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)

        x, y, w, h = self.prev_bbox
        margin = int(max(w, h) * 1.0)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        roi_diff = np.zeros_like(diff)
        roi_diff[y1:y2, x1:x2] = diff[y1:y2, x1:x2]

        cand = self._motion_bbox(roi_diff, frame.shape)

        if cand is None:
            self.lost_count += 1
            cand = self.prev_bbox
        else:
            self.lost_count = 0
            self.prev_bbox = cand

        self.prev_frame = gray
        self.trajectory.append(self.prev_bbox)

        return self.prev_bbox

    def get_metrics(self):
        return {"fps": self.fps, "lost": self.lost_count}