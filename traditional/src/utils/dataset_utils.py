import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Iterator


class OTB100Dataset:
    def __init__(self, dataset_root: str = "data/inputs/OTB100"):
        self.dataset_root = dataset_root
        self.sequences = self._list_sequences()

    def _list_sequences(self) -> List[str]:
        if not os.path.exists(self.dataset_root):
            return []
        return sorted([d for d in os.listdir(self.dataset_root) 
                       if os.path.isdir(os.path.join(self.dataset_root, d))])

    def get_sequence_path(self, sequence_name: str) -> str:
        return os.path.join(self.dataset_root, sequence_name)

    def get_image_paths(self, sequence_name: str) -> List[str]:
        seq_path = self.get_sequence_path(sequence_name)
        
        img_dir = os.path.join(seq_path, "img")
        if not os.path.exists(img_dir):
            img_dir = seq_path
        
        img_files = sorted([f for f in os.listdir(img_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))])
        return [os.path.join(img_dir, f) for f in img_files]

    def load_groundtruth(self, sequence_name: str) -> List[Tuple[int, int, int, int]]:
        seq_path = self.get_sequence_path(sequence_name)
        
        gt_file = os.path.join(seq_path, "groundtruth.txt")
        if not os.path.exists(gt_file):
            gt_file = os.path.join(seq_path, "groundtruth_rect.txt")
        
        if not os.path.exists(gt_file):
            return []
        
        bboxes = []
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        start_idx = 0
        if lines:
            first_line = lines[0].strip()
            if first_line and ',' not in first_line and '\t' not in first_line:
                parts = first_line.split()
                if len(parts) == 1:
                    try:
                        int(first_line)
                        start_idx = 1
                    except ValueError:
                        pass
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = []
            if ',' in line:
                parts = line.split(',')
            elif '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()
            
            if len(parts) >= 4:
                try:
                    x, y, w, h = int(float(parts[0])), int(float(parts[1])), \
                                  int(float(parts[2])), int(float(parts[3]))
                    bboxes.append((x, y, w, h))
                except ValueError:
                    continue
        
        return bboxes

    def get_sequence_length(self, sequence_name: str) -> int:
        return len(self.get_image_paths(sequence_name))

    def get_first_frame(self, sequence_name: str) -> Optional[np.ndarray]:
        img_paths = self.get_image_paths(sequence_name)
        if not img_paths:
            return None
        return cv2.imread(img_paths[0])

    def get_init_bbox(self, sequence_name: str) -> Optional[Tuple[int, int, int, int]]:
        gt_bboxes = self.load_groundtruth(sequence_name)
        if gt_bboxes:
            return gt_bboxes[0]
        return None

    def iterate_frames(self, sequence_name: str) -> Iterator[Tuple[int, np.ndarray]]:
        img_paths = self.get_image_paths(sequence_name)
        for idx, img_path in enumerate(img_paths):
            frame = cv2.imread(img_path)
            if frame is not None:
                yield idx, frame

    def load_sequence(self, sequence_name: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        frames = []
        img_paths = self.get_image_paths(sequence_name)
        for img_path in img_paths:
            frame = cv2.imread(img_path)
            if frame is not None:
                frames.append(frame)
        
        gt_bboxes = self.load_groundtruth(sequence_name)
        return frames, gt_bboxes

    def get_all_sequences(self) -> List[str]:
        return self.sequences

    def sequence_exists(self, sequence_name: str) -> bool:
        return sequence_name in self.sequences


def load_got_sequence(sequence_path: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    frames = []
    gt_bboxes = []
    
    img_files = sorted([f for f in os.listdir(sequence_path) 
                        if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    for f in img_files:
        frame = cv2.imread(os.path.join(sequence_path, f))
        if frame is not None:
            frames.append(frame)
    
    gt_file = os.path.join(sequence_path, "groundtruth.txt")
    if not os.path.exists(gt_file):
        gt_file = os.path.join(os.path.dirname(sequence_path), f"{os.path.basename(sequence_path)}.txt")
    
    if os.path.exists(gt_file):
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = []
            if ',' in line:
                parts = line.split(',')
            elif '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()
            
            if len(parts) >= 4:
                try:
                    x, y, w, h = int(float(parts[0])), int(float(parts[1])), \
                                  int(float(parts[2])), int(float(parts[3]))
                    gt_bboxes.append((x, y, w, h))
                except ValueError:
                    continue
    
    return frames, gt_bboxes


class VideoDataset:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.video_path)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_frame_count(self) -> int:
        if self.cap is None:
            self.open()
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        if self.cap is None:
            self.open()
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_size(self) -> Tuple[int, int]:
        if self.cap is None:
            self.open()
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def get_first_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            self.open()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        return frame if ret else None

    def iterate_frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        if self.cap is None:
            self.open()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def load_frames(self) -> List[np.ndarray]:
        frames = []
        for _, frame in self.iterate_frames():
            frames.append(frame)
        return frames