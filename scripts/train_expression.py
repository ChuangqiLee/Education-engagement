#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.training.trainer import train_from_config
from engagement.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if config.get("task") != "expression":
        raise ValueError("expression trainer requires task: expression")
    print(json.dumps(train_from_config(config, args.resume), indent=2))


if __name__ == "__main__":
    main()

