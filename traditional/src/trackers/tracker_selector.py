import numpy as np


class TrackerSelector:
    """
    轻量级tracker选择器：
    - 不融合，只选择最优bbox
    - 防止单个tracker崩溃污染结果
    """

    def __init__(self):
        self.prev_bbox = None
        self.prev_center = None
        self.prev_area = None

        self.motion_weight = 0.5
        self.size_weight = 0.3
        self.stability_weight = 0.2

    def _bbox_center(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2])

    def _bbox_area(self, bbox):
        return bbox[2] * bbox[3]

    def score(self, bbox):
        """
        越高越好
        """

        if bbox is None:
            return -1

        center = self._bbox_center(bbox)
        area = self._bbox_area(bbox)

        # 初始化情况
        if self.prev_center is None:
            return 1.0

        # 1️⃣ motion consistency
        motion_dist = np.linalg.norm(center - self.prev_center)
        motion_score = 1.0 / (1.0 + motion_dist / 50.0)

        # 2️⃣ size consistency
        size_change = abs(area - self.prev_area) / (self.prev_area + 1e-6) if self.prev_area else 0
        size_score = 1.0 / (1.0 + size_change)

        # 3️⃣ stability（中心不跳）
        stability_score = 1.0 / (1.0 + motion_dist / 100.0)

        score = (
            motion_score * self.motion_weight +
            size_score * self.size_weight +
            stability_score * self.stability_weight
        )

        return score

    def update(self, bbox):
        if bbox is None:
            return

        self.prev_bbox = bbox
        self.prev_center = self._bbox_center(bbox)
        self.prev_area = self._bbox_area(bbox)