"""OpenCV solvePnP head-pose estimator."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import (
    DEFAULT_MODEL_POINTS,
    approximate_camera_matrix,
    rotation_matrix_to_euler,
    select_pose_points,
)


@dataclass
class HeadPoseResult:
    pitch: float
    yaw: float
    roll: float
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    camera_matrix: np.ndarray

    def engagement_level(self, yaw_threshold: float = 23.6, pitch_threshold: float = 25.4) -> str:
        return "G3" if abs(self.yaw) > yaw_threshold or abs(self.pitch) > pitch_threshold else "G1"


class HeadPoseEstimator:
    def __init__(
        self,
        model_points: Optional[np.ndarray] = None,
        camera_matrix: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
    ):
        self.model_points = np.asarray(
            DEFAULT_MODEL_POINTS if model_points is None else model_points, dtype=np.float64
        )
        if self.model_points.shape != (6, 3):
            raise ValueError("model_points must have shape [6,3]")
        self.camera_matrix = None if camera_matrix is None else np.asarray(camera_matrix, dtype=np.float64)
        self.distortion = np.zeros((4, 1), dtype=np.float64) if distortion is None else np.asarray(distortion, dtype=np.float64)

    def estimate_from_points(self, image_points: np.ndarray, image_size: Tuple[int, int]) -> HeadPoseResult:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for solvePnP") from exc
        image_points = np.asarray(image_points, dtype=np.float64)
        if image_points.shape != (6, 2):
            raise ValueError("image_points must have shape [6,2]")
        if not np.isfinite(image_points).all():
            raise ValueError("image_points contain non-finite values")
        camera = self.camera_matrix
        if camera is None:
            camera = approximate_camera_matrix(image_size)
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera,
            self.distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise RuntimeError("OpenCV solvePnP did not converge")
        rotation, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = rotation_matrix_to_euler(rotation)
        return HeadPoseResult(pitch, yaw, roll, rvec, tvec, camera)

    def estimate_from_landmarks(self, landmarks: np.ndarray, image_size: Tuple[int, int]) -> HeadPoseResult:
        return self.estimate_from_points(select_pose_points(landmarks), image_size)

    def draw_axes(self, image_bgr: np.ndarray, result: HeadPoseResult, length: float = 120.0) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to draw pose axes") from exc
        axes = np.asarray(((0, 0, 0), (length, 0, 0), (0, length, 0), (0, 0, length)), dtype=np.float64)
        points, _ = cv2.projectPoints(
            axes, result.rotation_vector, result.translation_vector, result.camera_matrix, self.distortion
        )
        points = points.reshape(-1, 2).astype(int)
        canvas = image_bgr.copy()
        origin = tuple(points[0])
        cv2.line(canvas, origin, tuple(points[1]), (0, 0, 255), 2)
        cv2.line(canvas, origin, tuple(points[2]), (0, 255, 0), 2)
        cv2.line(canvas, origin, tuple(points[3]), (255, 0, 0), 2)
        return canvas

