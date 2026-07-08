import cv2
import numpy as np
from models.base_tracker import BaseTracker


class ColorThresholdTracker(BaseTracker):
    """
    颜色阈值法跟踪器（深度优化版）。
    核心改进：
    1. 中心区域初始化直方图（排除边框背景噪声）
    2. 自适应直方图bins（根据目标尺寸）
    3. 运动预测加速搜索
    4. 窗口退化恢复机制（CamShift窗口缩小时自动恢复）
    5. 模板匹配后备（当CamShift和轮廓都失败时）
    """
    def __init__(self, tracker_name="color"):
        super().__init__(tracker_name)

        self.roi_hist = None
        self.track_window = None
        self.prev_bbox = None
        self.target_area = 0
        self.target_aspect = 1.0
        self.hist_update_count = 0
        self.lost_count = 0

        # 形态学核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 运动速度（平滑）
        self.velocity = np.array([0.0, 0.0])
        # 原始初始化框（用于窗口退化恢复）
        self.init_bbox = None
        # 目标模板（灰度，用于模板匹配后备）
        self.target_template = None

    def init(self, frame, bbox):
        x, y, w, h = [int(v) for v in bbox]
        self.init_bbox = (x, y, w, h)
        self.prev_bbox = (x, y, w, h)
        self.target_area = w * h
        self.target_aspect = w / (h + 1e-6)

        # 中心区域采样，排除边框背景像素干扰
        margin_x, margin_y = int(w * 0.15), int(h * 0.15)
        y1 = max(0, y + margin_y)
        y2 = min(frame.shape[0], y + h - margin_y)
        x1 = max(0, x + margin_x)
        x2 = min(frame.shape[1], x + w - margin_x)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = hsv[y1:y2, x1:x2]

        if roi.size == 0:
            roi = hsv[y:y+h, x:x+w]

        # 排除低饱和度/低亮度像素后计算直方图
        mask = cv2.inRange(roi, (0, 30, 30), (180, 255, 255))

        # 自适应bins：根据目标大小调整直方图精度
        h_bins = min(60, max(30, w // 5))
        s_bins = min(64, max(32, h // 5))
        self.roi_hist = cv2.calcHist([roi], [0, 1], mask, [h_bins, s_bins], [0, 180, 0, 256])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.track_window = (x, y, w, h)

        # 保存灰度模板用于模板匹配后备
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.target_template = gray[y:y+h, x:x+w]

        self.trajectory = [bbox]
        self.frame_count = 1

    def _center(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2, y + h / 2])

    def _predict_window(self):
        """用运动预测估计当前窗口位置"""
        cx, cy = self._center(self.track_window)
        w, h = self.track_window[2], self.track_window[3]
        pred_cx = cx + self.velocity[0]
        pred_cy = cy + self.velocity[1]
        return (int(pred_cx - w/2), int(pred_cy - h/2), int(w), int(h))

    def _recover_window(self):
        """窗口退化恢复：如果窗口太小，恢复到初始大小但保持在当前位置"""
        x, y, w, h = self.track_window
        if w < self.init_bbox[2] * 0.3 or h < self.init_bbox[3] * 0.3:
            # 恢复大小
            rw, rh = self.init_bbox[2], self.init_bbox[3]
            cx, cy = x + w//2, y + h//2
            x_new = max(0, cx - rw//2)
            y_new = max(0, cy - rh//2)
            self.track_window = (x_new, y_new, rw, rh)

    def _search_by_camshift(self, frame, margin_scale):
        """CamShift跟踪（含运动预测初始化 + 窗口退化恢复）"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
        back_proj = cv2.GaussianBlur(back_proj, (5, 5), 0)

        # 用运动预测修正起始搜索位置
        pred_window = self._predict_window()
        x, y, w, h = pred_window
        h_img, w_img = frame.shape[:2]
        max_dim = max(w, h)

        margin = int(max_dim * margin_scale)
        x_search = max(0, x - margin)
        y_search = max(0, y - margin)
        w_search = min(w_img - x_search, w + 2 * margin)
        h_search = min(h_img - y_search, h + 2 * margin)

        search_window = (x_search, y_search, w_search, h_search)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1)

        try:
            ret, new_window = cv2.CamShift(back_proj, search_window, term_crit)
            x_new, y_new, w_new, h_new = new_window
            w_new = max(15, int(w_new))  # 比之前更高的最小值，防退化
            h_new = max(15, int(h_new))

            if w_new > 5 and h_new > 5:
                self.track_window = (x_new, y_new, w_new, h_new)
                # 窗口退化恢复
                self._recover_window()
                return (int(self.track_window[0]), int(self.track_window[1]),
                        int(self.track_window[2]), int(self.track_window[3]))
        except:
            pass

        return None

    def _search_by_contour(self, frame):
        """后备：用颜色概率+轮廓提取（改进评分）"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        prob = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
        prob = cv2.GaussianBlur(prob, (7, 7), 0)

        # 自适应阈值
        max_prob = prob.max()
        if max_prob < 10:
            return None

        _, thresh = cv2.threshold(prob, max(10, max_prob * 0.25), 255, cv2.THRESH_BINARY)
        thresh = thresh.astype(np.uint8)

        # 形态学清理
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        prev_center = self._center(self.prev_bbox)
        max_dim = max(self.prev_bbox[2], self.prev_bbox[3])
        best_box = None
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < max(50, self.target_area * 0.03) or area > self.target_area * 10:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / (h + 1e-6)
            if abs(aspect - self.target_aspect) > 6:
                continue

            center = np.array([x + w/2, y + h/2])
            dist = np.linalg.norm(center - prev_center)

            dist_score = 1.0 / (1.0 + dist / max_dim)
            area_score = min(area / self.target_area, self.target_area / area)
            aspect_score = 1.0 / (1.0 + abs(aspect - self.target_aspect))
            score = 0.5 * dist_score + 0.3 * area_score + 0.2 * aspect_score

            if score > best_score:
                best_score = score
                best_box = (x, y, w, h)

        return best_box

    def _search_by_template(self, frame):
        """最后手段：模板匹配"""
        if self.target_template is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = self.target_template.shape
        x, y, _, _ = self.prev_bbox

        # 扩大搜索
        max_dim = max(th, tw)
        margin = int(max_dim * (1.0 + self.lost_count * 0.5))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + tw + margin)
        y2 = min(frame.shape[0], y + th + margin)

        search_roi = gray[y1:y2, x1:x2]
        if search_roi.shape[0] < th or search_roi.shape[1] < tw:
            return None

        res = cv2.matchTemplate(search_roi, self.target_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.4:
            tx, ty = max_loc
            return (x1 + tx, y1 + ty, tw, th)
        return None

    def update(self, frame):
        # 第一步：多尺度CamShift
        margins = [1.0, 2.0, 4.0]
        if self.lost_count > 3:
            margins = [2.0, 4.0, 8.0]

        cand = None
        for margin in margins:
            cand = self._search_by_camshift(frame, margin)
            if cand is not None:
                break

        # 第二步：轮廓搜索
        if cand is None:
            cand = self._search_by_contour(frame)

        # 第三步：模板匹配（丢失较多时才用）
        if cand is None and self.lost_count > 2:
            cand = self._search_by_template(frame)

        if cand is None:
            cand = self.prev_bbox

        # 有效性检查
        prev_center = self._center(self.prev_bbox)
        new_center = self._center(cand)
        dist = np.linalg.norm(new_center - prev_center)
        max_dim = max(self.prev_bbox[2], self.prev_bbox[3])
        new_area = cand[2] * cand[3]
        area_ratio = new_area / (self.target_area + 1e-6)

        if dist < max_dim * 5 and 0.03 < area_ratio < 12:
            self.prev_bbox = cand
            self.lost_count = 0

            # 更新运动速度
            self.velocity = 0.6 * self.velocity + 0.4 * (new_center - prev_center)

            # 更新直方图（每5帧更新一次，更快适应变化）
            if self.hist_update_count % 5 == 0:
                x_new, y_new, w_new, h_new = cand
                hsv_new = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                roi_new = hsv_new[max(0,int(y_new)):min(frame.shape[0],int(y_new+h_new)),
                                 max(0,int(x_new)):min(frame.shape[1],int(x_new+w_new))]
                if roi_new.size > 100:
                    mask_new = cv2.inRange(roi_new, (0, 20, 20), (180, 255, 255))
                    new_hist = cv2.calcHist([roi_new], [0, 1], mask_new,
                                           [self.roi_hist.shape[0], self.roi_hist.shape[1]],
                                           [0, 180, 0, 256])
                    cv2.normalize(new_hist, new_hist, 0, 255, cv2.NORM_MINMAX)
                    self.roi_hist = 0.8 * self.roi_hist + 0.2 * new_hist
        else:
            self.lost_count += 1

        self.hist_update_count += 1
        self.frame_count += 1
        self.trajectory.append(self.prev_bbox)

        return self.prev_bbox

    def get_metrics(self):
        return {"fps": self.fps, "lost": self.lost_count}