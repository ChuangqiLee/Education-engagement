import numpy as np
import pytest

from engagement.landmarks.base import SyntheticLandmarkDetector
from engagement.landmarks.paper_stub import PaperLandmarkDetector


def test_synthetic_detector_has_468_points_and_multiple_faces():
    detector = SyntheticLandmarkDetector([(10, 10, 80, 90), (120, 20, 60, 70)])
    results = detector.detect(np.zeros((160, 220, 3), dtype=np.uint8))
    assert len(results) == 2
    assert all(result.landmarks.shape == (468, 3) for result in results)
    assert all(result.provenance == "SYNTHETIC_SMOKE_TEST" for result in results)


def test_paper_detector_fails_honestly():
    with pytest.raises(NotImplementedError, match="UNRESOLVED"):
        PaperLandmarkDetector().detect(np.zeros((32, 32, 3), dtype=np.uint8))

