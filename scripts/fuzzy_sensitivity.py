#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engagement.fuzzy.evaluator import FuzzyEngagementEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep primary fuzzy weights; demo assumptions only")
    parser.add_argument("--output", default="outputs/statistics/fuzzy_sensitivity.csv")
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    rows = []
    n = int(round(1 / args.step))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            w1, w2 = a * args.step, b * args.step
            w3 = max(0.0, 1.0 - w1 - w2)
            evaluator = FuzzyEngagementEvaluator((w1, w2, w3))
            result = evaluator.evaluate_predictions(
                {"positive": 0.7, "neutral": 0.2, "negative": 0.1},
                pitch=10,
                yaw=28,
                behavior_probabilities={"nodding": 0.1, "writing": 0.55, "turn_head": 0.2, "phone": 0.1, "sleeping": 0.05},
            )
            rows.append((w1, w2, w3, result.final_score, result.grade))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("facial_weight", "head_weight", "body_weight", "demo_score", "grade"))
        writer.writerows(rows)
    print("wrote %d ASSUMPTION-based scenarios to %s" % (len(rows), output))


if __name__ == "__main__":
    main()
