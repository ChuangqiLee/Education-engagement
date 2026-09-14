"""Replaceable centroid tracker baseline ([ASSUMPTION])."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


Box = Tuple[int, int, int, int]


@dataclass
class _Track:
    centroid: np.ndarray
    missing: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_missing: int = 8):
        self.max_distance = float(max_distance)
        self.max_missing = int(max_missing)
        self.tracks: Dict[int, _Track] = {}
        self.next_id = 1

    @staticmethod
    def _centroid(box: Box) -> np.ndarray:
        x, y, width, height = box
        return np.asarray((x + width / 2.0, y + height / 2.0), dtype=float)

    def update(self, boxes: Sequence[Box]) -> List[int]:
        centroids = [self._centroid(box) for box in boxes]
        assignments = [-1] * len(boxes)
        available_tracks = set(self.tracks)
        available_boxes = set(range(len(boxes)))
        candidates = []
        for track_id in available_tracks:
            for box_index in available_boxes:
                distance = float(np.linalg.norm(self.tracks[track_id].centroid - centroids[box_index]))
                candidates.append((distance, track_id, box_index))
        for distance, track_id, box_index in sorted(candidates):
            if distance > self.max_distance or track_id not in available_tracks or box_index not in available_boxes:
                continue
            assignments[box_index] = track_id
            self.tracks[track_id] = _Track(centroids[box_index], 0)
            available_tracks.remove(track_id)
            available_boxes.remove(box_index)
        for box_index in available_boxes:
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = _Track(centroids[box_index], 0)
            assignments[box_index] = track_id
        for track_id in list(available_tracks):
            self.tracks[track_id].missing += 1
            if self.tracks[track_id].missing > self.max_missing:
                del self.tracks[track_id]
        return assignments

