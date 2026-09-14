"""Fuzzy comprehensive evaluation (paper Equations 6-13)."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from engagement.utils.config import load_yaml

from .membership import (
    behavior_matrix,
    facial_matrix,
    head_pose_matrix,
    normalize,
    pose_activation_weights,
    validate_weights,
)
from .rules import GRADE_LABELS


@dataclass
class FuzzyResult:
    expression_membership: np.ndarray
    head_pose_membership: np.ndarray
    behavior_membership: np.ndarray
    membership: np.ndarray
    final_score: Optional[float]
    grade: str
    participation_label: str


class FuzzyEngagementEvaluator:
    def __init__(
        self,
        primary_weights: Sequence[float] = (0.4, 0.3, 0.3),
        score_vector: Optional[Sequence[float]] = (100.0, 60.0, 20.0),
        yaw_threshold: float = 23.6,
        pitch_threshold: float = 25.4,
    ):
        self.primary_weights = validate_weights(primary_weights, 3, "primary_weights")
        self.score_vector = None if score_vector is None else np.asarray(score_vector, dtype=np.float64)
        if self.score_vector is not None and self.score_vector.shape != (3,):
            raise ValueError("score_vector must have shape [3]")
        self.yaw_threshold = float(yaw_threshold)
        self.pitch_threshold = float(pitch_threshold)

    @classmethod
    def from_yaml(cls, path: str):
        config = load_yaml(path)
        thresholds = config.get("thresholds", {})
        return cls(
            primary_weights=config["primary_weights"],
            score_vector=config.get("score_vector"),
            yaw_threshold=thresholds.get("yaw_or_torsion_degrees", 23.6),
            pitch_threshold=thresholds.get("pitch_degrees", 25.4),
        )

    def evaluate_matrices(
        self,
        r11: np.ndarray,
        r12: np.ndarray,
        r13: np.ndarray,
        w1: Sequence[float],
        w2: Sequence[float],
        w3: Sequence[float],
    ) -> FuzzyResult:
        matrices = [np.asarray(r11, dtype=float), np.asarray(r12, dtype=float), np.asarray(r13, dtype=float)]
        expected_rows = (3, 4, 5)
        for index, (matrix, rows) in enumerate(zip(matrices, expected_rows), 1):
            if matrix.shape != (rows, 3):
                raise ValueError("R1%d must have shape [%d,3]" % (index, rows))
            if (matrix < 0).any() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("membership matrix rows must be nonnegative and sum to 1")
        weights = [
            validate_weights(w1, 3, "W1"),
            validate_weights(w2, 4, "W2"),
            validate_weights(w3, 5, "W3"),
        ]
        modality = [weight @ matrix for weight, matrix in zip(weights, matrices)]
        combined = self.primary_weights @ np.vstack(modality)
        combined = normalize(combined, "combined membership")
        grade = ("G1", "G2", "G3")[int(np.argmax(combined))]
        score = None if self.score_vector is None else float(combined @ self.score_vector)
        return FuzzyResult(modality[0], modality[1], modality[2], combined, score, grade, GRADE_LABELS[grade])

    def evaluate_predictions(
        self,
        expression_probabilities: Mapping[str, float],
        pitch: float,
        yaw: float,
        behavior_probabilities: Mapping[str, float],
    ) -> FuzzyResult:
        expression_order = ("positive", "neutral", "negative")
        behavior_order = ("nodding", "writing", "turn_head", "phone", "sleeping")
        w1 = normalize([expression_probabilities.get(label, 0.0) for label in expression_order], "expression probabilities")
        w2 = pose_activation_weights(pitch, yaw, self.pitch_threshold, self.yaw_threshold)
        w3 = normalize([behavior_probabilities.get(label, 0.0) for label in behavior_order], "behavior probabilities")
        return self.evaluate_matrices(facial_matrix(), head_pose_matrix(), behavior_matrix(), w1, w2, w3)

