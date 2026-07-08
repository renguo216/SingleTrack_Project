import os
import cv2
import numpy as np
import time

from models.traditional.frame_diff_tracker import FrameDiffTracker
from models.traditional.background_subtraction_tracker import BackgroundSubtractionTracker
from models.traditional.color_threshold_tracker import ColorThresholdTracker
from models.traditional.edge_contour_tracker import EdgeContourTracker

from utils.dataset_utils import load_got_sequence
from utils.evaluation import Evaluation


class TraditionalEvaluator:

    def __init__(self):

        self.trackers = {
            "frame_diff": FrameDiffTracker(),
            "bg_subtract": BackgroundSubtractionTracker(),
            "color": ColorThresholdTracker(),
            "edge": EdgeContourTracker()
        }
        self.evaluator = Evaluation()

    def evaluate_sequence(self, frames, gt_boxes):

        results = {
            "frame_diff": [],
            "bg_subtract": [],
            "color": [],
            "edge": []
        }

        start_time = time.time()

        first_frame = frames[0]
        init_box = gt_boxes[0]

        # init each tracker
        for t in self.trackers.values():
            t.init(first_frame, init_box)

        for i, frame in enumerate(frames):

            gt = gt_boxes[i]

            for name, tracker in self.trackers.items():

                bbox = tracker.update(frame)

                iou = self.evaluator.calculate_iou(bbox, gt)
                cle = self.evaluator.calculate_center_error(bbox, gt)

                results[name].append([iou, cle])

        end_time = time.time()
        fps = len(frames) / (end_time - start_time)

        return results, fps


def evaluate_dataset(test_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    sequence_list = os.listdir(test_dir)

    all_summary = []

    for seq in sequence_list:

        frames, gt_boxes = load_got_sequence(os.path.join(test_dir, seq))

        evaluator = TraditionalEvaluator()

        results, fps = evaluator.evaluate_sequence(frames, gt_boxes)

        print(f"\nSequence: {seq}")

        for name in results:

            ious = [x[0] for x in results[name]]
            cels = [x[1] for x in results[name]]

            avg_iou = np.mean(ious)
            avg_cle = np.mean(cels)

            print(f"{name}: IoU={avg_iou:.4f}, CLE={avg_cle:.2f}")

            all_summary.append([seq, name, avg_iou, avg_cle, fps])

    # save csv
    import csv

    csv_path = os.path.join(output_dir, "summary.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "method", "iou", "cle", "fps"])
        writer.writerows(all_summary)

    print("\nSaved:", csv_path)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    evaluate_dataset(args.test_dir, args.output_dir)