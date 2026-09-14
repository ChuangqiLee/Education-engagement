"""Honest placeholder for the undisclosed paper landmark detector."""

import numpy as np

from .base import LandmarkDetector


class PaperLandmarkDetector(LandmarkDetector):
    def detect(self, image_bgr: np.ndarray):
        raise NotImplementedError(
            "[UNRESOLVED] The paper attributes 468-point detection to XGBoost "
            "but gives no targets, features, topology, training data, or recoverable implementation. "
            "Use mediapipe_facemesh for the explicitly documented engineering substitute."
        )

