"""MediaPipe Face Mesh engineering substitute for the paper's 468 points."""

from typing import List

import numpy as np

from .base import LandmarkDetector, LandmarkResult


class MediaPipeFaceMeshDetector(LandmarkDetector):
    provenance = (
        "ASSUMPTION: MediaPipe operationalizes the paper's 468-landmark description; "
        "the paper does not specify MediaPipe."
    )

    def __init__(self, max_num_faces: int = 20, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe backend requested but mediapipe is not installed. "
                "Install the 'landmarks' extra or use the synthetic smoke backend."
            ) from exc
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This MediaPipe build lacks the legacy solutions.face_mesh API. "
                "Install a compatible mediapipe 0.10 build or add a Tasks adapter."
            )
        try:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "MediaPipe FaceMesh could not initialize. On macOS the legacy graph may "
                "require an OpenGL context even for CPU inference; run it in a graphical "
                "session or provide a MediaPipe Tasks CPU adapter. Original error: %s" % exc
            ) from exc

    def detect(self, image_bgr: np.ndarray) -> List[LandmarkResult]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for MediaPipe color conversion") from exc
        height, width = image_bgr.shape[:2]
        output = self._mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        results: List[LandmarkResult] = []
        for face in output.multi_face_landmarks or []:
            points = np.asarray(
                [(item.x * width, item.y * height, item.z * width) for item in face.landmark[:468]],
                dtype=np.float32,
            )
            if points.shape[0] != 468:
                continue
            xmin, ymin = np.floor(points[:, :2].min(axis=0)).astype(int)
            xmax, ymax = np.ceil(points[:, :2].max(axis=0)).astype(int)
            box = (max(0, xmin), max(0, ymin), max(1, xmax - xmin), max(1, ymax - ymin))
            results.append(LandmarkResult(points, box, 1.0, "mediapipe_facemesh", self.provenance))
        return results

    def close(self) -> None:
        self._mesh.close()


def build_landmark_detector(name: str):
    name = name.lower()
    if name == "paper_stub":
        from .paper_stub import PaperLandmarkDetector

        return PaperLandmarkDetector()
    if name == "mediapipe_facemesh":
        return MediaPipeFaceMeshDetector()
    if name == "synthetic":
        from .base import SyntheticLandmarkDetector

        return SyntheticLandmarkDetector()
    raise KeyError("unknown landmark backend: %s" % name)
