"""Paper label aliases and engagement-grade rules."""

from typing import Dict


EXPRESSION_ALIASES: Dict[str, str] = {
    "happy": "positive",
    "happiness": "positive",
    "positive": "positive",
    "serious": "neutral",
    "confusion": "neutral",
    "confused": "neutral",
    "neutral": "neutral",
    "bored": "negative",
    "boredom": "negative",
    "negative": "negative",
}

EXPRESSION_GRADES = {"positive": "G1", "neutral": "G2", "negative": "G3"}
BEHAVIOR_GRADES = {
    "nodding": "G1",
    "writing": "G1",
    "turn_head": "G2",
    "phone": "G3",
    "sleeping": "G3",
}
GRADE_LABELS = {"G1": "active", "G2": "partial", "G3": "passive"}


def canonical_expression(label: str) -> str:
    key = label.lower().strip().replace(" ", "_")
    if key not in EXPRESSION_ALIASES:
        raise KeyError("unknown expression label: %s" % label)
    return EXPRESSION_ALIASES[key]


def head_pose_grade(pitch: float, yaw_or_torsion: float, pitch_threshold: float = 25.4, yaw_threshold: float = 23.6) -> str:
    return "G3" if abs(pitch) > pitch_threshold or abs(yaw_or_torsion) > yaw_threshold else "G1"

