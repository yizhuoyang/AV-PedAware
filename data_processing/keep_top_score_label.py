#!/usr/bin/env python3
"""Keep only the highest-score inferred bbox for each unlabeled bag frame."""

import argparse
from pathlib import Path


LABELED_BAGS = {
    "rosbag2_2026_05_17-09_34_07",
    "rosbag2_2026_05_17-09_36_02",
    "rosbag2_2026_05_17-09_51_28",
    "rosbag2_2026_05_17-09_53_59",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("data/pairs"),
        help="Root containing bag directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned changes without rewriting labels.",
    )
    return parser.parse_args()


def iter_infer_bags(pairs_root):
    return sorted(
        path for path in pairs_root.iterdir()
        if path.is_dir() and path.name not in LABELED_BAGS
    )


def read_scored_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 16:
            raise ValueError("Expected 16 fields in {}, got {}".format(path, len(parts)))
        rows.append((" ".join(parts[:15]), float(parts[15])))
    return rows


def main():
    args = parse_args()
    changed = 0
    already_single = 0
    empty = 0

    for bag_dir in iter_infer_bags(args.pairs_root):
        scored_dir = bag_dir / "labels_with_scores"
        label_dir = bag_dir / "labels"
        for scored_path in sorted(scored_dir.glob("*.txt")):
            rows = read_scored_lines(scored_path)
            if len(rows) == 0:
                empty += 1
                continue
            if len(rows) == 1:
                already_single += 1
            else:
                changed += 1

            best_line, _ = max(rows, key=lambda row: row[1])
            if not args.dry_run:
                label_dir.mkdir(parents=True, exist_ok=True)
                (label_dir / scored_path.name).write_text(best_line + "\n")

    print("multi_box_frames_fixed: {}".format(changed))
    print("already_single_frames: {}".format(already_single))
    print("empty_frames_unchanged: {}".format(empty))


if __name__ == "__main__":
    main()
