"""Uniform facial landmark API.

The paper does not disclose a recoverable 468-point detector. Concrete
backends therefore carry explicit provenance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LandmarkResult:
    landmarks: np.ndarray
    bbox: Tuple[int, int, int, int]
    confidence: float
    backend: str
    provenance: str

    def __post_init__(self) -> None:
        points = np.asarray(self.landmarks, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("landmarks must have shape [N,2] or [N,3]")
        self.landmarks = points


class LandmarkDetector(ABC):
    expected_points = 468

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> List[LandmarkResult]:
        """Return zero or more faces in pixel coordinates."""

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SyntheticLandmarkDetector(LandmarkDetector):
    """Deterministic smoke-test mesh; never a scientific detector."""

    def __init__(self, face_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None):
        self.face_boxes = list(face_boxes) if face_boxes is not None else None

    @staticmethod
    def _mesh(box: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> np.ndarray:
        x, y, w, h = box
        index = np.arange(468, dtype=np.float32)
        angle = index * (np.pi * (3.0 - np.sqrt(5.0)))
        radius = np.sqrt((index + 0.5) / 468.0)
        px = x + w * (0.5 + 0.43 * radius * np.cos(angle))
        py = y + h * (0.5 + 0.48 * radius * np.sin(angle))
        pz = 0.04 * w * np.cos(angle * 0.5)
        mesh = np.stack((px, py, pz), axis=1).astype(np.float32)
        # Give the six PnP indices an exact zero-rotation projection so the
        # smoke pipeline validates pose plumbing without fake pose changes.
        from engagement.head_pose.geometry import DEFAULT_MODEL_POINTS, MEDIAPIPE_POSE_INDICES

        image_width, image_height = image_size
        focal = float(image_width)
        tz = float(max(w, h) * 10)
        target_x, target_y = x + w / 2.0, y + h / 2.0
        tx = (target_x - image_width / 2.0) * tz / focal
        ty = (target_y - image_height / 2.0) * tz / focal
        translated = DEFAULT_MODEL_POINTS + np.asarray((tx, ty, tz))
        projected = np.column_stack(
            (
                focal * translated[:, 0] / translated[:, 2] + image_width / 2.0,
                focal * translated[:, 1] / translated[:, 2] + image_height / 2.0,
            )
        )
        for idx, point in zip(MEDIAPIPE_POSE_INDICES, projected):
            mesh[idx, :2] = point
        return mesh

    def detect(self, image_bgr: np.ndarray) -> List[LandmarkResult]:
        height, width = image_bgr.shape[:2]
        boxes = self.face_boxes or [(width // 4, height // 6, width // 2, int(height * 0.68))]
        return [
            LandmarkResult(self._mesh(box, (width, height)), box, 1.0, "synthetic", "SYNTHETIC_SMOKE_TEST")
            for box in boxes
        ]


def draw_landmarks(image_bgr: np.ndarray, results: Sequence[LandmarkResult], step: int = 4) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for landmark visualization") from exc
    canvas = image_bgr.copy()
    for result in results:
        for x, y in result.landmarks[:: max(1, step), :2]:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (80, 255, 120), -1)
        x, y, w, h = result.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 255, 120), 1)
        cv2.putText(canvas, result.backend, (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 255, 120), 1)
    return canvas
