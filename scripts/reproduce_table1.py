#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy paper Table 1 reference values without claiming reproduction")
    parser.add_argument("--output-dir", default="outputs/model_comparison")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile("paper_reference_results.csv", output / "paper_reference_results.csv")
    actual = output / "actual_experiment_results.csv"
    if not actual.exists():
        with actual.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(("run_name", "seed", "dataset_version", "test_accuracy", "checkpoint", "provenance"))
    print("paper reference copied to %s; actual results remain separate at %s" % (output / "paper_reference_results.csv", actual))


if __name__ == "__main__":
    main()

