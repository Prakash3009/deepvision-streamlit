# sort.py  — Simple SORT Tracker (stable version)
# Works for people entering/exiting line

import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.no_losses = 0

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0

    def get_state(self):
        return self.bbox


class Sort:
    def __init__(self, max_age=5, min_hits=2, iou_threshold=0.1):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
        return iou

    def update(self, dets):
        if len(self.trackers) == 0:
            for d in dets:
                self.trackers.append(KalmanBoxTracker(d))
            return [(t.get_state(), t.id) for t in self.trackers]

        # Compute IOU
        iou_matrix = np.zeros((len(self.trackers), len(dets)), dtype=np.float32)
        for t, trk in enumerate(self.trackers):
            for d, det in enumerate(dets):
                iou_matrix[t, d] = self.iou(trk.get_state(), det)

        matched = []
        unmatched_trackers = []
        unmatched_detections = []

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_trackers.append(r)
                unmatched_detections.append(c)
            else:
                matched.append((r, c))

        for t in range(len(self.trackers)):
            if t not in row_ind:
                unmatched_trackers.append(t)

        for d in range(len(dets)):
            if d not in col_ind:
                unmatched_detections.append(d)

        # Update matched
        for r, c in matched:
            self.trackers[r].update(dets[c])

        # Create new trackers
        for idx in unmatched_detections:
            self.trackers.append(KalmanBoxTracker(dets[idx]))

        # Remove old trackers
        to_remove = []
        for t, trk in enumerate(self.trackers):
            trk.no_losses += 1
            if trk.no_losses > self.max_age:
                to_remove.append(t)

        for idx in reversed(to_remove):
            self.trackers.pop(idx)

        return [(t.get_state(), t.id) for t in self.trackers]
