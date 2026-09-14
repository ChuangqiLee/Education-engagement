"""PnP geometry and Euler-angle conversion."""

from typing import Optional, Sequence, Tuple

import numpy as np


DEFAULT_MODEL_POINTS = np.asarray(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, 330.0, -65.0),      # chin; +Y follows image-down convention
        (-225.0, -170.0, -135.0), # left eye outer
        (225.0, -170.0, -135.0),  # right eye outer
        (-150.0, 150.0, -125.0),  # left mouth
        (150.0, 150.0, -125.0),   # right mouth
    ],
    dtype=np.float64,
)

MEDIAPIPE_POSE_INDICES = (1, 152, 33, 263, 61, 291)


def approximate_camera_matrix(image_size: Tuple[int, int], focal_length: Optional[float] = None) -> np.ndarray:
    """[ASSUMPTION] focal length equals image width; principal point is center."""
    width, height = image_size
    focal = float(width if focal_length is None else focal_length)
    return np.asarray(((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)))


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """Return intrinsic x/y/z rotations as pitch, yaw, roll in degrees."""
    matrix = np.asarray(rotation_matrix, dtype=np.float64)
    sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    singular = sy < 1e-8
    if not singular:
        pitch = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        pitch = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = 0.0
    return tuple(float(np.degrees(item)) for item in (pitch, yaw, roll))


def select_pose_points(landmarks: np.ndarray, indices: Sequence[int] = MEDIAPIPE_POSE_INDICES) -> np.ndarray:
    points = np.asarray(landmarks)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("landmarks must have shape [N,2+] for pose estimation")
    if points.shape[0] <= max(indices):
        raise ValueError("at least %d indexed landmarks are required" % (max(indices) + 1))
    selected = points[list(indices), :2].astype(np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("pose landmarks contain non-finite values")
    return selected
