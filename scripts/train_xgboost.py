#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from engagement.utils.config import load_yaml
from engagement.xgb.model import LandmarkXGBoost


def main() -> None:
    parser = argparse.ArgumentParser(description="Train practical XGBoost auxiliary classifier")
    parser.add_argument("--config", default="configs/xgboost_paper.yaml")
    parser.add_argument("--csv", help="CSV whose last column is the label")
    parser.add_argument("--output", default="checkpoints/xgboost_demo.joblib")
    parser.add_argument("--synthetic", action="store_true", help="Use labelled synthetic tabular data for smoke testing")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.csv:
        data = np.genfromtxt(args.csv, delimiter=",", skip_header=1)
        x, y = data[:, :-1], data[:, -1]
        provenance = "USER_CSV"
    elif args.synthetic:
        from sklearn.datasets import make_classification

        x, y = make_classification(n_samples=80, n_features=24, n_informative=12, n_classes=3, random_state=42)
        provenance = "SYNTHETIC_SMOKE_TEST"
    else:
        parser.error("provide --csv or explicitly opt into --synthetic")
    ignored = {"mode", "objective"}
    params = {k: v for k, v in config.items() if k not in ignored}
    model = LandmarkXGBoost(config.get("mode", "practical_reproduction"), **params).fit(x, y)
    model.save(args.output)
    print(json.dumps({"provenance": provenance, "metrics_on_training_data": model.evaluate(x, y), "checkpoint": args.output}, indent=2))


if __name__ == "__main__":
    main()

