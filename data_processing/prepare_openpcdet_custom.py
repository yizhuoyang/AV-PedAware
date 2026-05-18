#!/usr/bin/env python3
"""Prepare labeled AV-PedAware lidar frames for OpenPCDet custom training."""

import argparse
import shutil
from pathlib import Path

import numpy as np


DEFAULT_SPLITS = {
    "train": [
        "rosbag2_2026_05_17-09_34_07",
        "rosbag2_2026_05_17-09_36_02",
        "rosbag2_2026_05_17-09_51_28",
        "rosbag2_2026_05_17-09_53_59",
    ],
    "val": [],
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
        "--output-root",
        type=Path,
        default=Path("OpenPCDet/data/custom"),
        help="OpenPCDet custom dataset output root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing custom dataset directory.",
    )
    return parser.parse_args()


def prepare_output_dirs(output_root, overwrite):
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    for name in ("points", "labels", "ImageSets"):
        (output_root / name).mkdir(parents=True, exist_ok=True)


def convert_labelcloud_kitti_line(line):
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) != 15:
        raise ValueError("Expected 15 KITTI-style fields, got {}: {}".format(len(parts), line))

    class_name = parts[0].lower()
    if class_name not in {"person", "pedestrian"}:
        return None

    # labelCloud kitti_untransformed export uses:
    # class ... h w l x y z yaw
    h, w, l = map(float, parts[8:11])
    x, y, z, yaw = map(float, parts[11:15])
    return "{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} Pedestrian".format(
        x, y, z, l, w, h, yaw
    )


def convert_split(pairs_root, output_root, split, bag_names, start_index):
    sample_ids = []
    next_index = start_index

    for bag_name in bag_names:
        bag_root = pairs_root / bag_name
        lidar_dir = bag_root / "lidar"
        label_dir = bag_root / "labels"
        if not lidar_dir.exists() or not label_dir.exists():
            raise FileNotFoundError("Missing lidar or labels directory for {}".format(bag_name))

        for lidar_path in sorted(lidar_dir.glob("*.bin")):
            label_path = label_dir / "{}.txt".format(lidar_path.stem)
            if not label_path.exists():
                continue

            converted = []
            for line in label_path.read_text().splitlines():
                new_line = convert_labelcloud_kitti_line(line)
                if new_line is not None:
                    converted.append(new_line)
            if not converted:
                continue

            sample_id = "{:06d}".format(next_index)
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            np.save(output_root / "points" / "{}.npy".format(sample_id), points)
            (output_root / "labels" / "{}.txt".format(sample_id)).write_text(
                "\n".join(converted) + "\n"
            )

            sample_ids.append(sample_id)
            next_index += 1

    (output_root / "ImageSets" / "{}.txt".format(split)).write_text(
        "\n".join(sample_ids) + ("\n" if sample_ids else "")
    )
    return next_index, len(sample_ids)


def main():
    args = parse_args()
    prepare_output_dirs(args.output_root, args.overwrite)

    next_index = 1
    counts = {}
    for split in ("train", "val"):
        next_index, count = convert_split(
            args.pairs_root,
            args.output_root,
            split,
            DEFAULT_SPLITS[split],
            next_index,
        )
        counts[split] = count

    print("train_samples: {}".format(counts["train"]))
    print("val_samples: {}".format(counts["val"]))
    print("output_root: {}".format(args.output_root))


if __name__ == "__main__":
    main()
