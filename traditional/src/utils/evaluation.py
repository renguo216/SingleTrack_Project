import numpy as np
from typing import List, Tuple, Dict


def calc_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calc_cle(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    center1_x = x1 + w1 / 2
    center1_y = y1 + h1 / 2
    center2_x = x2 + w2 / 2
    center2_y = y2 + h2 / 2
    
    return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


class Evaluation:
    def __init__(self, iou_threshold: float = 0.5, 
                 center_error_threshold: float = 20.0):
        self.iou_threshold = iou_threshold
        self.center_error_threshold = center_error_threshold

    def calculate_iou(self, box1: Tuple[int, int, int, int],
                      box2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
        x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def calculate_center_error(self, box1: Tuple[int, int, int, int],
                               box2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        error = np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)
        return error

    def evaluate_sequence(self, pred_bboxes: List[Tuple[int, int, int, int]],
                          gt_bboxes: List[Tuple[int, int, int, int]]) -> Dict:
        if len(pred_bboxes) != len(gt_bboxes):
            raise ValueError("Prediction and ground truth length mismatch")
        
        ious = []
        center_errors = []
        success_count = 0
        
        for pred, gt in zip(pred_bboxes, gt_bboxes):
            iou = self.calculate_iou(pred, gt)
            center_error = self.calculate_center_error(pred, gt)
            
            ious.append(iou)
            center_errors.append(center_error)
            
            if iou >= self.iou_threshold:
                success_count += 1
        
        total_frames = len(pred_bboxes)
        
        precision = len([e for e in center_errors if e <= self.center_error_threshold]) / total_frames
        success_rate = success_count / total_frames
        average_iou = np.mean(ious)
        average_center_error = np.mean(center_errors)
        
        sorted_ious = np.sort(ious)[::-1]
        auc = np.trapz(sorted_ious, np.arange(1, total_frames + 1)) / total_frames
        
        return {
            'precision': precision,
            'success': success_rate,
            'auc': auc,
            'average_iou': average_iou,
            'average_center_error': average_center_error,
            'iou_list': ious,
            'center_error_list': center_errors,
            'total_frames': total_frames,
            'success_frames': success_count
        }

    def evaluate(self, pred_bboxes: List[List[int]],
                 gt_bboxes: List[List[int]]) -> Dict:
        pred_tuples = [tuple(b) for b in pred_bboxes]
        gt_tuples = [tuple(b) for b in gt_bboxes]
        
        return self.evaluate_sequence(pred_tuples, gt_tuples)

    def generate_success_plot(self, ious: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        thresholds = np.linspace(0, 1, 21)
        success_rates = []
        
        for threshold in thresholds:
            success_count = len([iou for iou in ious if iou >= threshold])
            success_rate = success_count / len(ious)
            success_rates.append(success_rate)
        
        return thresholds, np.array(success_rates)

    def generate_precision_plot(self, center_errors: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        thresholds = np.arange(0, 51, 1)
        precision_rates = []
        
        for threshold in thresholds:
            count = len([e for e in center_errors if e <= threshold])
            precision_rate = count / len(center_errors)
            precision_rates.append(precision_rate)
        
        return thresholds, np.array(precision_rates)