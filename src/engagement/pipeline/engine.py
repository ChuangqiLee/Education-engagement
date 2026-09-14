"""End-to-end multimodal engagement pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engagement.fuzzy.evaluator import FuzzyEngagementEvaluator
from engagement.head_pose.pnp import HeadPoseEstimator
from engagement.landmarks.mediapipe_backend import build_landmark_detector
from engagement.models.classifier import DemoUntrainedClassifier, TorchImageClassifier
from engagement.tracking.centroid import CentroidTracker
from engagement.utils.config import load_yaml
from engagement.xgb.features import landmark_features
from engagement.xgb.model import LandmarkXGBoost


EXPRESSION_CLASSES = ("positive", "neutral", "negative")
BEHAVIOR_CLASSES = ("nodding", "writing", "turn_head", "phone", "sleeping")


@dataclass
class PipelineRecord:
    timestamp: float
    source_id: str
    track_id: int
    bbox: Tuple[int, int, int, int]
    expression: str
    expression_confidence: float
    body_behavior: str
    body_confidence: float
    pitch: float
    yaw: float
    roll: float
    xgboost_output: Optional[str]
    fuzzy_membership: List[float]
    engagement_score: Optional[float]
    engagement_level: str
    participation_label: str
    provenance: str


class EngagementPipeline:
    def __init__(
        self,
        landmark_detector,
        expression_classifier,
        behavior_classifier,
        pose_estimator: Optional[HeadPoseEstimator] = None,
        fuzzy_evaluator: Optional[FuzzyEngagementEvaluator] = None,
        tracker: Optional[CentroidTracker] = None,
        xgboost_model: Optional[LandmarkXGBoost] = None,
        demo: bool = False,
    ):
        self.landmark_detector = landmark_detector
        self.expression_classifier = expression_classifier
        self.behavior_classifier = behavior_classifier
        self.pose_estimator = pose_estimator or HeadPoseEstimator()
        self.fuzzy_evaluator = fuzzy_evaluator or FuzzyEngagementEvaluator()
        self.tracker = tracker or CentroidTracker()
        self.xgboost_model = xgboost_model
        self.demo = bool(demo)

    @classmethod
    def from_config(cls, path: str):
        config = load_yaml(path)
        demo = bool(config.get("demo", False))
        detector = build_landmark_detector(config["landmarks"]["backend"])
        model_config = config.get("models", {})
        if demo:
            expression = DemoUntrainedClassifier(EXPRESSION_CLASSES, offset=0)
            behavior = DemoUntrainedClassifier(BEHAVIOR_CLASSES, offset=1)
        else:
            expression_checkpoint = model_config.get("checkpoint_expression")
            behavior_checkpoint = model_config.get("checkpoint_behavior")
            if not expression_checkpoint or not behavior_checkpoint:
                raise RuntimeError(
                    "trained expression and behavior checkpoints are required outside demo mode; "
                    "use configs/pipeline_demo.yaml for a clearly labelled smoke run"
                )
            expression = TorchImageClassifier(model_config["expression"], EXPRESSION_CLASSES, expression_checkpoint)
            behavior = TorchImageClassifier(model_config["behavior"], BEHAVIOR_CLASSES, behavior_checkpoint)
        fuzzy_path = config["fuzzy"]["config"]
        fuzzy = FuzzyEngagementEvaluator.from_yaml(fuzzy_path)
        tracking = config.get("tracking", {})
        tracker = CentroidTracker(
            tracking.get("max_distance_pixels", 80), tracking.get("max_missing_frames", 8)
        )
        xgb_model = None
        checkpoint = config.get("xgboost", {}).get("checkpoint")
        if checkpoint:
            xgb_model = LandmarkXGBoost.load(checkpoint)
        return cls(detector, expression, behavior, fuzzy_evaluator=fuzzy, tracker=tracker, xgboost_model=xgb_model, demo=demo)

    @staticmethod
    def _crop(image: np.ndarray, box: Tuple[int, int, int, int], expansion: float = 0.15) -> np.ndarray:
        x, y, width, height = box
        pad_x, pad_y = int(width * expansion), int(height * expansion)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(image.shape[1], x + width + pad_x), min(image.shape[0], y + height + pad_y)
        crop = image[y0:y1, x0:x1]
        return image if crop.size == 0 else crop

    def process_frame(self, image_bgr: np.ndarray, timestamp: float = 0.0, source_id: str = "image") -> List[PipelineRecord]:
        faces = self.landmark_detector.detect(image_bgr)
        track_ids = self.tracker.update([face.bbox for face in faces])
        records = []
        image_size = (image_bgr.shape[1], image_bgr.shape[0])
        for face, track_id in zip(faces, track_ids):
            face_crop = self._crop(image_bgr, face.bbox, 0.1)
            body_crop = self._crop(image_bgr, face.bbox, 0.8)
            expression, expression_confidence, expression_probabilities = self.expression_classifier.predict(face_crop)
            behavior, body_confidence, behavior_probabilities = self.behavior_classifier.predict(body_crop)
            try:
                pose = self.pose_estimator.estimate_from_landmarks(face.landmarks, image_size)
                pitch, yaw, roll = pose.pitch, pose.yaw, pose.roll
            except Exception:
                if not self.demo:
                    raise
                pitch = yaw = roll = 0.0
            xgb_output = None
            if self.xgboost_model is not None:
                xgb_output = str(self.xgboost_model.predict(landmark_features(face.landmarks)[None, :])[0])
            fuzzy = self.fuzzy_evaluator.evaluate_predictions(
                expression_probabilities, pitch, yaw, behavior_probabilities
            )
            records.append(
                PipelineRecord(
                    timestamp=float(timestamp),
                    source_id=source_id,
                    track_id=track_id,
                    bbox=face.bbox,
                    expression=expression,
                    expression_confidence=float(expression_confidence),
                    body_behavior=behavior,
                    body_confidence=float(body_confidence),
                    pitch=float(pitch),
                    yaw=float(yaw),
                    roll=float(roll),
                    xgboost_output=xgb_output,
                    fuzzy_membership=fuzzy.membership.tolist(),
                    engagement_score=fuzzy.final_score,
                    engagement_level=fuzzy.grade,
                    participation_label=fuzzy.participation_label,
                    provenance="DEMO_UNTRAINED" if self.demo else "MODEL_INFERENCE",
                )
            )
        return records

    def close(self) -> None:
        self.landmark_detector.close()

