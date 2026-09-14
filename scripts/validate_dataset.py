#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.data.synthetic import BEHAVIOR_CLASSES, EXPRESSION_CLASSES
from engagement.data.validate import validate_imagefolder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--task", choices=("expression", "body_behavior"), required=True)
    args = parser.parse_args()
    classes = EXPRESSION_CLASSES if args.task == "expression" else BEHAVIOR_CLASSES
    report = validate_imagefolder(args.root, classes)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()

