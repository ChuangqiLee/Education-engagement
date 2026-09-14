import numpy as np
import pytest

from engagement.fuzzy.evaluator import FuzzyEngagementEvaluator
from engagement.fuzzy.membership import behavior_matrix, facial_matrix, head_pose_matrix
from engagement.fuzzy.rules import canonical_expression


def test_aliases_retain_paper_wording():
    assert canonical_expression("happy") == "positive"
    assert canonical_expression("serious") == "neutral"
    assert canonical_expression("confusion") == "neutral"
    assert canonical_expression("bored") == "negative"


def test_matrix_shapes_and_weight_validation():
    assert facial_matrix().shape == (3, 3)
    assert head_pose_matrix().shape == (4, 3)
    assert behavior_matrix().shape == (5, 3)
    evaluator = FuzzyEngagementEvaluator()
    with pytest.raises(ValueError, match="sum to 1"):
        evaluator.evaluate_matrices(facial_matrix(), head_pose_matrix(), behavior_matrix(), [1, 1, 1], [0.25] * 4, [0.2] * 5)


def test_active_and_passive_predictions():
    evaluator = FuzzyEngagementEvaluator()
    active = evaluator.evaluate_predictions(
        {"positive": 1, "neutral": 0, "negative": 0},
        pitch=0,
        yaw=0,
        behavior_probabilities={"nodding": 0, "writing": 1, "turn_head": 0, "phone": 0, "sleeping": 0},
    )
    assert active.grade == "G1"
    assert np.isclose(active.membership.sum(), 1)
    passive = evaluator.evaluate_predictions(
        {"positive": 0, "neutral": 0, "negative": 1},
        pitch=30,
        yaw=30,
        behavior_probabilities={"nodding": 0, "writing": 0, "turn_head": 0, "phone": 1, "sleeping": 0},
    )
    assert passive.grade == "G3"
    assert passive.final_score < active.final_score


def test_paper_config_leaves_score_unresolved():
    evaluator = FuzzyEngagementEvaluator.from_yaml("configs/fuzzy_paper.yaml")
    result = evaluator.evaluate_predictions(
        {"positive": 1, "neutral": 0, "negative": 0}, 0, 0,
        {"nodding": 1, "writing": 0, "turn_head": 0, "phone": 0, "sleeping": 0},
    )
    assert result.final_score is None

