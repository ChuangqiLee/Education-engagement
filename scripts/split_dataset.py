#!/usr/bin/env python3
"""Create leakage-aware train/val/test folders from a manifest.

Manifest columns: path,label,group_id. All items with one group_id remain in
one split, preventing adjacent frames or one student from crossing partitions.
"""

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        parser.error("fractions must be nonnegative and sum to less than one")
    grouped = defaultdict(list)
    with open(args.manifest, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["group_id"]].append(row)
    group_ids = sorted(grouped)
    random.Random(args.seed).shuffle(group_ids)
    test_count = round(len(group_ids) * args.test_fraction)
    val_count = round(len(group_ids) * args.val_fraction)
    assignments = {}
    for index, group_id in enumerate(group_ids):
        assignments[group_id] = "test" if index < test_count else "val" if index < test_count + val_count else "train"
    output = Path(args.output)
    seen_targets = set()
    for group_id, rows in grouped.items():
        split = assignments[group_id]
        for row in rows:
            source = Path(row["path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            target = output / split / row["label"] / (group_id + "__" + source.name)
            if target in seen_targets:
                raise ValueError("duplicate output filename: %s" % target)
            seen_targets.add(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    with (output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("group_id", "split"))
        writer.writerows(sorted(assignments.items()))
    print("split %d groups into %s" % (len(group_ids), output))


if __name__ == "__main__":
    main()

