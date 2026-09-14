"""Membership matrices matching Equations 6-8 and paper label rules."""

from typing import Mapping, Sequence, Tuple

import numpy as np

from .rules import BEHAVIOR_GRADES, EXPRESSION_GRADES


GRADES = ("G1", "G2", "G3")


def normalize(values: Sequence[float], name: str = "weights") -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("%s must be a finite one-dimensional vector" % name)
    if (vector < 0).any():
        raise ValueError("%s cannot contain negative values" % name)
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("%s must have a positive sum" % name)
    return vector / total


def validate_weights(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (expected,):
        raise ValueError("%s must have shape [%d]" % (name, expected))
    if not np.isclose(vector.sum(), 1.0, atol=1e-6):
        raise ValueError("%s must sum to 1" % name)
    if (vector < 0).any():
        raise ValueError("%s cannot contain negative values" % name)
    return vector


def grade_matrix(labels: Sequence[str], mapping: Mapping[str, str]) -> np.ndarray:
    matrix = np.zeros((len(labels), 3), dtype=np.float64)
    for row, label in enumerate(labels):
        grade = mapping[label]
        matrix[row, GRADES.index(grade)] = 1.0
    return matrix


def facial_matrix() -> np.ndarray:
    """R11: rows positive/neutral/negative; columns G1/G2/G3."""
    return grade_matrix(("positive", "neutral", "negative"), EXPRESSION_GRADES)


def behavior_matrix() -> np.ndarray:
    """R13: rows nodding/writing/turn_head/phone/sleeping."""
    return grade_matrix(("nodding", "writing", "turn_head", "phone", "sleeping"), BEHAVIOR_GRADES)


def head_pose_matrix() -> np.ndarray:
    """R12: rows yaw-in/yaw-out/pitch-in/pitch-out.

    Table 5 only assigns G1 or G3; it does not define a partial/G2 pose state.
    """
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def pose_activation_weights(pitch: float, yaw: float, pitch_threshold: float, yaw_threshold: float) -> np.ndarray:
    return np.asarray(
        (
            0.5 if abs(yaw) <= yaw_threshold else 0.0,
            0.5 if abs(yaw) > yaw_threshold else 0.0,
            0.5 if abs(pitch) <= pitch_threshold else 0.0,
            0.5 if abs(pitch) > pitch_threshold else 0.0,
        )
    )

