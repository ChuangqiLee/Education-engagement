import numpy as np
import pytest

from engagement.xgb.features import landmark_features
from engagement.xgb.model import LandmarkXGBoost, PAPER_PARAMS


pytest.importorskip("xgboost")


def test_paper_parameters_are_preserved():
    assert PAPER_PARAMS == {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
    }


def test_landmark_features_and_tiny_fit(tmp_path):
    rng = np.random.default_rng(42)
    mesh = rng.normal(size=(468, 3)).astype(np.float32)
    feature = landmark_features(mesh)
    assert feature.ndim == 1 and feature.size > 1404
    x = rng.normal(size=(36, 12))
    y = np.repeat(("G1", "G2", "G3"), 12)
    model = LandmarkXGBoost(n_estimators=4, max_depth=2, n_jobs=1).fit(x, y)
    assert model.predict(x[:3]).shape == (3,)
    assert model.predict_proba(x[:3]).shape == (3, 3)
    checkpoint = tmp_path / "tiny.joblib"
    model.save(checkpoint)
    assert LandmarkXGBoost.load(checkpoint).predict(x[:2]).shape == (2,)


def test_paper_faithful_mode_is_explicitly_unresolved():
    with pytest.raises(NotImplementedError, match="UNRESOLVED"):
        LandmarkXGBoost(mode="paper_faithful").fit(np.zeros((2, 2)), (0, 1))

