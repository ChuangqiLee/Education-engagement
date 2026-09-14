#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_groups(path: str):
    groups = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row["group"], []).append(float(row["score"]))
    if len(groups) != 2:
        raise ValueError("raw CSV must contain exactly two group values")
    return groups, "USER_RAW_DATA"


def synthetic_groups(seed: int = 42):
    import numpy as np

    rng = np.random.default_rng(seed)
    return {
        "Class 1": rng.normal(83.84, 9.29336, 116),
        "Class 2": rng.normal(81.21, 9.50577, 118),
    }, "SYNTHETIC_PIPELINE_DEMO"


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-score tests; synthetic mode validates code only")
    parser.add_argument("--csv", help="student_id,group,score CSV")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/statistics/raw_test_results.json")
    args = parser.parse_args()
    if bool(args.csv) == bool(args.synthetic):
        parser.error("provide exactly one of --csv or --synthetic")
    groups, provenance = load_groups(args.csv) if args.csv else synthetic_groups(args.seed)
    import numpy as np
    from scipy import stats

    names = list(groups)
    first, second = np.asarray(groups[names[0]]), np.asarray(groups[names[1]])
    result = {
        "provenance": provenance,
        "warning": "Synthetic results are not a reproduction of paper p-values." if args.synthetic else None,
        "groups": {name: {"n": len(values), "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))} for name, values in groups.items()},
        "kstest_standardized": {
            name: dict(zip(("statistic", "pvalue"), map(float, stats.kstest((np.asarray(values) - np.mean(values)) / np.std(values, ddof=1), "norm"))))
            for name, values in groups.items()
        },
        "ks_note": "SciPy one-sample K-S after estimated standardization is not identical to SPSS's corrected normality test.",
        "levene": dict(zip(("statistic", "pvalue"), map(float, stats.levene(first, second)))),
        "ttest_equal": dict(zip(("statistic", "pvalue"), map(float, stats.ttest_ind(first, second, equal_var=True)))),
        "ttest_welch": dict(zip(("statistic", "pvalue"), map(float, stats.ttest_ind(first, second, equal_var=False)))),
        "anova": dict(zip(("statistic", "pvalue"), map(float, stats.f_oneway(first, second)))),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

