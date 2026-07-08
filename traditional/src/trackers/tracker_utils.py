import numpy as np

class BBoxValidator:
    def __init__(self,
                 max_area_ratio=2.5,
                 min_area_ratio=0.3,
                 max_center_shift=80,
                 max_aspect_change=0.6):

        self.max_area_ratio = max_area_ratio
        self.min_area_ratio = min_area_ratio
        self.max_center_shift = max_center_shift
        self.max_aspect_change = max_aspect_change

    def center(self, b):
        x, y, w, h = b
        return np.array([x + w / 2, y + h / 2])

    def area(self, b):
        return b[2] * b[3]

    def aspect(self, b):
        return b[2] / (b[3] + 1e-6)

    def validate(self, prev_bbox, new_bbox):
        if prev_bbox is None or new_bbox is None:
            return True

        prev_area = self.area(prev_bbox)
        new_area = self.area(new_bbox)

        if prev_area <= 0:
            return True

        area_ratio = new_area / (prev_area + 1e-6)

        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            return False

        prev_c = self.center(prev_bbox)
        new_c = self.center(new_bbox)

        shift = np.linalg.norm(prev_c - new_c)

        if shift > self.max_center_shift:
            return False

        prev_aspect = self.aspect(prev_bbox)
        new_aspect = self.aspect(new_bbox)

        if abs(prev_aspect - new_aspect) > self.max_aspect_change:
            return False

        return True


class ConfidenceFusion:
    def compute(self, motion_score=0, template_score=0, shape_score=0, distance_score=0):
        return (
            0.35 * motion_score +
            0.35 * template_score +
            0.2 * shape_score +
            0.1 * distance_score
        )


class TrackerState:
    TRACKING = 0
    UNCERTAIN = 1
    LOST = 2