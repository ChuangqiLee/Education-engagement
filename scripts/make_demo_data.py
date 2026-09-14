#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.data.synthetic import make_demo_datasets, make_demo_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labelled synthetic smoke-test assets")
    parser.add_argument("--root", default="data/demo")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    counts = make_demo_datasets(args.root, args.samples_per_class, args.size)
    video = make_demo_video(Path(args.root) / "synthetic_classroom.mp4")
    print(json.dumps({"counts": counts, "video": str(video), "provenance": "SYNTHETIC"}, indent=2))


if __name__ == "__main__":
    main()
