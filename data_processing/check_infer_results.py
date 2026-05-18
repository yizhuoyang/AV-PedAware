#!/usr/bin/env python3
"""Check inferred labels and render lidar+bbox previews for AV-PedAware bags."""

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
        help="Root containing bag_name/lidar and bag_name/labels.",
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
        "--include-labeled-bags",
        action="store_true",
        help="Include manually labeled training bags instead of checking only inferred bags.",
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


def render_frame(points, boxes, output_path):
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

    ax.set_title("{} boxes".format(len(boxes)))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-4.1, 4.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="z (m)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def iter_bags(pairs_root, include_labeled_bags):
    return sorted(
        path for path in pairs_root.iterdir()
        if path.is_dir() and (include_labeled_bags or path.name not in LABELED_BAGS)
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "infer_check_report.csv"

    total_frames = 0
    problem_frames = 0
    with report_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bag", "sample", "num_boxes", "status"])

        for bag_dir in iter_bags(args.pairs_root, args.include_labeled_bags):
            for lidar_path in sorted((bag_dir / "lidar").glob("*.bin")):
                sample = lidar_path.stem
                boxes = parse_labelcloud_kitti(bag_dir / "labels" / "{}.txt".format(sample))
                status = "ok" if len(boxes) == 1 else "problem"
                writer.writerow([bag_dir.name, sample, len(boxes), status])
                total_frames += 1
                problem_frames += int(status == "problem")

                should_render = args.render and (not args.only_problems or status == "problem")
                if should_render:
                    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
                    render_frame(
                        points,
                        boxes,
                        args.output_dir / bag_dir.name / "{}.png".format(sample),
                    )

    print("report: {}".format(report_path))
    print("total_frames: {}".format(total_frames))
    print("problem_frames: {}".format(problem_frames))
    print("all_single_target: {}".format(problem_frames == 0))


if __name__ == "__main__":
    main()
