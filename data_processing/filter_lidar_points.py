#!/usr/bin/env python3
"""Level and filter paired lidar point clouds by XY radius and Z height."""

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("data/pairs"),
        help="Root directory containing bag_name/lidar/*.bin files.",
    )
    parser.add_argument(
        "--xy-radius",
        type=float,
        default=5.0,
        help="Keep points within this XY radius from the origin.",
    )
    parser.add_argument(
        "--max-z",
        type=float,
        default=1.9,
        help="Keep points with z lower than or equal to this value.",
    )
    parser.add_argument(
        "--pitch-degrees",
        type=float,
        default=0.0,
        help="Rotate points around +Y before filtering to compensate lidar pitch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report filtering statistics without rewriting files.",
    )
    return parser.parse_args()


def iter_lidar_files(pairs_root):
    return sorted(pairs_root.glob("*/lidar/*.bin"))


def filter_points(points, xy_radius, max_z):
    xy_sq = points[:, 0] ** 2 + points[:, 1] ** 2
    keep = (xy_sq <= xy_radius ** 2) & (points[:, 2] <= max_z)
    return points[keep]


def rotate_points_y(points, pitch_degrees):
    if pitch_degrees == 0:
        return points
    angle = np.deg2rad(pitch_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation = np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float32,
    )
    rotated = points.copy()
    rotated[:, :3] = points[:, :3] @ rotation.T
    return rotated


def main():
    args = parse_args()
    files = iter_lidar_files(args.pairs_root)
    total_before = 0
    total_after = 0

    for path in files:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        leveled = rotate_points_y(points, args.pitch_degrees)
        filtered = filter_points(leveled, args.xy_radius, args.max_z)
        total_before += len(points)
        total_after += len(filtered)
        if not args.dry_run:
            filtered.astype(np.float32, copy=False).tofile(path)

    print("files: {}".format(len(files)))
    print("points_before: {}".format(total_before))
    print("points_after: {}".format(total_after))
    print("points_removed: {}".format(total_before - total_after))


if __name__ == "__main__":
    main()
