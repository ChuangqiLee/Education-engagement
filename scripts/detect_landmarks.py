#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.io.video_source import read_image
from engagement.landmarks.base import draw_landmarks
from engagement.landmarks.mediapipe_backend import build_landmark_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="468-point detector smoke/operational CLI")
    parser.add_argument("image")
    parser.add_argument("--backend", choices=("paper_stub", "mediapipe_facemesh", "synthetic"), default="mediapipe_facemesh")
    parser.add_argument("--output", default="outputs/smoke/landmark_overlay.png")
    args = parser.parse_args()
    import cv2

    image = read_image(args.image)
    detector = build_landmark_detector(args.backend)
    try:
        results = detector.detect(image)
    finally:
        detector.close()
    overlay = draw_landmarks(image, results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise RuntimeError("failed to write overlay")
    print(json.dumps({"faces": len(results), "point_counts": [len(item.landmarks) for item in results], "output": str(output), "backend": args.backend}, indent=2))


if __name__ == "__main__":
    main()

