"""Clearly labelled synthetic assets for smoke tests only."""

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

from PIL import Image, ImageDraw


EXPRESSION_CLASSES = ("positive", "neutral", "negative")
BEHAVIOR_CLASSES = ("nodding", "writing", "turn_head", "phone", "sleeping")


def _pattern(label: str, index: int, size: int) -> Image.Image:
    palette = {
        "positive": (72, 180, 105),
        "neutral": (120, 130, 145),
        "negative": (180, 78, 78),
        "nodding": (70, 135, 190),
        "writing": (150, 95, 185),
        "turn_head": (218, 150, 55),
        "phone": (180, 70, 145),
        "sleeping": (65, 90, 150),
    }
    image = Image.new("RGB", (size, size), palette[label])
    draw = ImageDraw.Draw(image)
    cx = size // 2 + int(math.sin(index) * size * 0.08)
    cy = size // 2 + int(math.cos(index) * size * 0.08)
    r = size // 4
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline="white", width=max(1, size // 32))
    draw.line((0, (index * 13) % size, size, (index * 29) % size), fill="white", width=2)
    draw.rectangle((3, 3, size - 4, 19), fill="black")
    draw.text((6, 5), "SYNTHETIC " + label, fill="white")
    return image


def make_imagefolder(
    root: Union[str, Path],
    classes: Sequence[str],
    samples_per_class: int = 8,
    size: int = 64,
    splits: Sequence[str] = ("train", "val", "test"),
) -> Dict[str, int]:
    root = Path(root)
    counts: Dict[str, int] = {}
    split_offsets = {name: position * 1000 for position, name in enumerate(splits)}
    for split in splits:
        for label in classes:
            target = root / split / label
            target.mkdir(parents=True, exist_ok=True)
            split_count = samples_per_class if split == "train" else max(2, samples_per_class // 3)
            for index in range(split_count):
                _pattern(label, index + split_offsets[split], size).save(target / ("synthetic_%03d.png" % index))
            counts["%s/%s" % (split, label)] = split_count
    marker = root / "SYNTHETIC_DATA_ONLY.txt"
    marker.write_text(
        "Generated procedural images for smoke testing. These are not paper data and must not be used as research results.\n",
        encoding="utf-8",
    )
    return counts


def make_demo_datasets(root: Union[str, Path] = "data/demo", samples_per_class: int = 8, size: int = 256) -> Dict[str, int]:
    root = Path(root)
    counts = make_imagefolder(root / "expression", EXPRESSION_CLASSES, samples_per_class, size=size)
    counts.update(
        {"body_behavior/" + k: v for k, v in make_imagefolder(root / "body_behavior", BEHAVIOR_CLASSES, samples_per_class, size=size).items()}
    )
    return counts


def make_demo_video(path: Union[str, Path], frames: int = 18, fps: float = 3.0, size=(320, 240)) -> Path:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required to generate demo video") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError("could not create demo video at %s" % path)
    width, height = size
    for index in range(frames):
        frame = np.full((height, width, 3), (35, 45, 65), dtype=np.uint8)
        center = (40 + (index * 12) % (width - 80), height // 2)
        cv2.circle(frame, center, 34, (80, 175, 235), -1)
        cv2.putText(frame, "SYNTHETIC / DEMO", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "frame %d" % index, (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()
    return path
