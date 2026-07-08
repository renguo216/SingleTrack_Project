from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


class BaseTracker(ABC):
    def __init__(self, tracker_name: str):
        self.tracker_name = tracker_name
        self.trajectory: List[Tuple[int, int, int, int]] = []
        self.frame_count = 0
        self.fps = 0.0
        self.init_bbox: Optional[Tuple[int, int, int, int]] = None

    @abstractmethod
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        pass

    @abstractmethod
    def update(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        pass

    def reset(self) -> None:
        self.trajectory = []
        self.frame_count = 0
        self.fps = 0.0
        self.init_bbox = None

    def get_trajectory(self) -> List[Tuple[int, int, int, int]]:
        return self.trajectory

    def get_tracker_name(self) -> str:
        return self.tracker_name