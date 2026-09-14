"""Render per-student engagement annotations."""

from typing import Iterable

import numpy as np


def draw_records(image_bgr: np.ndarray, records: Iterable[object], demo: bool = False) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for overlays") from exc
    canvas = image_bgr.copy()
    if demo:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (10, 10, 170), -1)
        banner_scale = min(0.55, max(0.28, canvas.shape[1] / 900.0))
        cv2.putText(canvas, "DEMO / UNTRAINED - NOT A RESEARCH RESULT", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, banner_scale, (255, 255, 255), 1, cv2.LINE_AA)
    colors = {"G1": (60, 190, 80), "G2": (40, 190, 230), "G3": (50, 70, 230)}
    for record in records:
        x, y, width, height = record.bbox
        color = colors.get(record.engagement_level, (220, 220, 220))
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2)
        lines = [
            "ID %s %s/%s" % (record.track_id, record.engagement_level, record.participation_label),
            "%s %.2f | %s %.2f" % (record.expression, record.expression_confidence, record.body_behavior, record.body_confidence),
            "P %.1f Y %.1f R %.1f | score %s" % (
                record.pitch,
                record.yaw,
                record.roll,
                "NA" if record.engagement_score is None else "%.1f" % record.engagement_score,
            ),
        ]
        text_scale = 0.42 if canvas.shape[1] >= 480 else 0.31
        text_x = min(x, 5)
        text_start = max(48, y + 14)
        last_baseline = min(canvas.shape[0] - 4, text_start + (len(lines) - 1) * 15)
        cv2.rectangle(
            canvas,
            (text_x, max(33, text_start - 11)),
            (canvas.shape[1] - 2, min(canvas.shape[0] - 1, last_baseline + 4)),
            (12, 12, 12),
            -1,
        )
        for index, line in enumerate(lines):
            baseline = min(canvas.shape[0] - 4, text_start + index * 15)
            cv2.putText(canvas, line, (text_x + 2, baseline), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, 1, cv2.LINE_AA)
    return canvas
