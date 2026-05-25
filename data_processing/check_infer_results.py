#!/usr/bin/env python3
"""Check inferred labels and render lidar+bbox previews for pairs datasets."""

import argparse
import csv
from pathlib import Path

import numpy as np


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
        help="Root recursively containing sequence/lidar and sequence/labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/infer_checks"),
        help="Directory for reports and rendered previews.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render lidar+bbox PNG previews.",
    )
    parser.add_argument(
        "--only-problems",
        action="store_true",
        help="Render only frames that are not single-target predictions.",
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Optional needs_review.csv from infer_ros1_pairs.py.",
    )
    parser.add_argument(
        "--only-review",
        action="store_true",
        help="Render only frames listed in --review-csv.",
    )
    parser.add_argument(
        "--exclude-seqs",
        nargs="*",
        default=[],
        help="Sequence directory names to exclude, for example person11 person23.",
    )
    parser.add_argument(
        "--axis-limit",
        type=float,
        default=5.3,
        help="Symmetric XY plot limit in meters.",
    )
    parser.add_argument(
        "--include-labeled-bags",
        action="store_true",
        help="Include legacy manually labeled ROS 2 bags and names passed with --exclude-seqs.",
    )
    return parser.parse_args()


def parse_labelcloud_kitti(path):
    if not path.exists():
        return np.empty((0, 7), dtype=np.float32)

    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 15:
            raise ValueError("Expected 15 fields in {}, got {}".format(path, len(parts)))
        h, w, l = map(float, parts[8:11])
        x, y, z, yaw = map(float, parts[11:15])
        boxes.append([x, y, z, l, w, h, yaw])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 7)


def render_frame(points, boxes, output_path, title, axis_limit):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        s=0.35,
        cmap="viridis",
        linewidths=0,
    )

    for box in boxes:
        x, y, _, dx, dy, _, yaw = box
        corners = np.array(
            [
                [dx / 2, dy / 2],
                [dx / 2, -dy / 2],
                [-dx / 2, -dy / 2],
                [-dx / 2, dy / 2],
            ],
            dtype=np.float32,
        )
        rotation = np.array(
            [
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)],
            ],
            dtype=np.float32,
        )
        corners = corners @ rotation.T + np.array([x, y], dtype=np.float32)
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
            )
        )

    ax.set_title("{} | {} boxes".format(title, len(boxes)))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="z (m)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_review_status(path):
    if path is None:
        return {}
    with path.open(newline="") as handle:
        return {
            (row["sequence"], row["sample"]): row["status"]
            for row in csv.DictReader(handle)
        }


def iter_bags(pairs_root, include_labeled_bags, exclude_seqs):
    if (pairs_root / "lidar").is_dir():
        candidates = [pairs_root]
    else:
        candidates = sorted(path.parent for path in pairs_root.rglob("lidar") if path.is_dir())
    excluded = set(exclude_seqs)
    if not include_labeled_bags:
        excluded.update(LABELED_BAGS)
    return [path for path in candidates if path.name not in excluded]


def main():
    args = parse_args()
    if args.only_review and args.review_csv is None:
        raise ValueError("--only-review requires --review-csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "infer_check_report.csv"
    review_status = load_review_status(args.review_csv)

    total_frames = 0
    problem_frames = 0
    review_frames = 0
    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "sample", "num_boxes", "status", "review_status"])

        for bag_dir in iter_bags(args.pairs_root, args.include_labeled_bags, args.exclude_seqs):
            sequence = str(bag_dir.relative_to(args.pairs_root))
            for lidar_path in sorted((bag_dir / "lidar").glob("*.bin")):
                sample = lidar_path.stem
                boxes = parse_labelcloud_kitti(bag_dir / "labels" / "{}.txt".format(sample))
                status = "ok" if len(boxes) == 1 else "problem"
                flagged_status = review_status.get((sequence, sample), "")
                writer.writerow([sequence, sample, len(boxes), status, flagged_status])
                total_frames += 1
                problem_frames += int(status == "problem")
                review_frames += int(bool(flagged_status))

                should_render = args.render
                if args.only_problems:
                    should_render = should_render and status == "problem"
                if args.only_review:
                    should_render = should_render and bool(flagged_status)
                if should_render:
                    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
                    title = "{}/{}".format(sequence, sample)
                    if flagged_status:
                        title += " | {}".format(flagged_status)
                    render_frame(
                        points,
                        boxes,
                        args.output_dir / sequence / "{}.png".format(sample),
                        title,
                        args.axis_limit,
                    )

    print("report: {}".format(report_path))
    print("total_frames: {}".format(total_frames))
    print("problem_frames: {}".format(problem_frames))
    print("flagged_review_frames: {}".format(review_frames))
    print("all_single_target: {}".format(problem_frames == 0))


if __name__ == "__main__":
    main()
