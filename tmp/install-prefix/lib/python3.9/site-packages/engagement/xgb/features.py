"""Practical tabular features derived from an already detected face mesh.

This module is an [ASSUMPTION], not a reconstruction of the undisclosed
XGBoost keypoint detector described by the paper.
"""

from typing import Iterable, List, Sequence

import numpy as np


ANGLE_TRIPLES = ((33, 1, 263), (61, 1, 291), (10, 1, 152), (33, 168, 263))
DISTANCE_PAIRS = ((33, 263), (61, 291), (1, 152), (10, 152), (33, 61), (263, 291))
SYMMETRY_PAIRS = ((33, 263), (61, 291), (127, 356), (234, 454))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    left, right = a - b, c - b
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] != 468 or points.shape[1] not in (2, 3):
        raise ValueError("expected landmarks with shape [468,2] or [468,3]")
    minimum = points[:, :2].min(axis=0)
    span = np.maximum(points[:, :2].max(axis=0) - minimum, 1e-6)
    xy = (points[:, :2] - minimum) / span
    if points.shape[1] == 2:
        z = np.zeros((468, 1), dtype=np.float32)
    else:
        scale = float(max(span))
        z = (points[:, 2:3] - points[:, 2:3].mean()) / max(scale, 1e-6)
    return np.concatenate((xy, z), axis=1)


def landmark_features(landmarks: np.ndarray, include_coordinates: bool = True) -> np.ndarray:
    points = normalize_landmarks(landmarks)
    features: List[float] = []
    if include_coordinates:
        features.extend(points.reshape(-1).tolist())
    for first, second in DISTANCE_PAIRS:
        features.append(float(np.linalg.norm(points[first] - points[second])))
    for first, vertex, third in ANGLE_TRIPLES:
        features.append(_angle(points[first], points[vertex], points[third]))
    center_x = float(points[1, 0])
    for left, right in SYMMETRY_PAIRS:
        features.append(abs(abs(float(points[left, 0]) - center_x) - abs(float(points[right, 0]) - center_x)))
    for axis in range(3):
        values = points[:, axis]
        features.extend(
            [float(values.mean()), float(values.std()), float(np.quantile(values, 0.25)), float(np.median(values)), float(np.quantile(values, 0.75))]
        )
    return np.asarray(features, dtype=np.float32)


def batch_landmark_features(meshes: Iterable[np.ndarray], include_coordinates: bool = True) -> np.ndarray:
    rows = [landmark_features(mesh, include_coordinates) for mesh in meshes]
    if not rows:
        raise ValueError("at least one landmark mesh is required")
    return np.stack(rows)

