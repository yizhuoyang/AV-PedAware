#!/usr/bin/env python3
"""Visualize an xyzi float32 lidar .bin file with Open3D or matplotlib."""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_LIDAR = Path("data/pairs_ros1/train/clock11/lidar/0001.bin")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "lidar_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LIDAR,
        help="Path to a float32 x/y/z/intensity .bin file.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "open3d", "matplotlib"],
        default="auto",
        help="Visualization backend. auto tries Open3D first and falls back to matplotlib.",
    )
    parser.add_argument(
        "--color-by",
        choices=["intensity", "z"],
        default="intensity",
        help="Scalar used to color the points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Randomly display at most this many points; zero displays all points.",
    )
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size.")
    parser.add_argument("--save", type=Path, help="Save a matplotlib bird's-eye-view PNG preview.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window. Useful with --save.",
    )
    return parser.parse_args()


def load_xyzi(path):
    if not path.exists():
        raise FileNotFoundError("LiDAR file does not exist: {}".format(path))
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(
            "{} contains {} float32 values, which cannot be reshaped as xyzi points".format(
                path, values.size
            )
        )
    points = values.reshape(-1, 4)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("No valid points in: {}".format(path))
    return points


def sample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(0)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def point_colors(points, color_by):
    values = points[:, 3] if color_by == "intensity" else points[:, 2]
    low, high = np.percentile(values, [2.0, 98.0])
    if high <= low:
        normalized = np.zeros_like(values)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    from matplotlib import colormaps

    return colormaps["viridis"](normalized)[:, :3]


def print_stats(path, points):
    xyz_min = points[:, :3].min(axis=0)
    xyz_max = points[:, :3].max(axis=0)
    intensity = points[:, 3]
    print("file: {}".format(path))
    print("points: {}".format(len(points)))
    print(
        "xyz min: [{:.3f}, {:.3f}, {:.3f}]  max: [{:.3f}, {:.3f}, {:.3f}]".format(
            xyz_min[0], xyz_min[1], xyz_min[2], xyz_max[0], xyz_max[1], xyz_max[2]
        )
    )
    print("intensity min/max: {:.3f} / {:.3f}".format(intensity.min(), intensity.max()))


def show_open3d(points, colors, point_size):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    cloud.colors = o3d.utility.Vector3dVector(colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    visualizer = o3d.visualization.Visualizer()
    created = visualizer.create_window(window_name="LiDAR point cloud", width=1280, height=800)
    if not created:
        raise RuntimeError("Open3D failed to create a visualization window")
    visualizer.add_geometry(cloud)
    visualizer.add_geometry(axis)
    options = visualizer.get_render_option()
    options.point_size = point_size
    options.background_color = np.asarray([0.03, 0.03, 0.03])
    visualizer.run()
    visualizer.destroy_window()


def matplotlib_preview(points, colors, lidar_path, output_path, show):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), dpi=160)
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=0.7, linewidths=0)
    ax.scatter([0.0], [0.0], marker="+", s=80, c="red", linewidths=1.2)
    ax.set_title("LiDAR bird's-eye view: {}".format(lidar_path.name))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print("preview: {}".format(output_path))
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    points = load_xyzi(args.lidar_path)
    print_stats(args.lidar_path, points)
    shown_points = sample_points(points, args.max_points)
    colors = point_colors(shown_points, args.color_by)

    if args.save:
        matplotlib_preview(shown_points, colors, args.lidar_path, args.save, show=False)

    if args.no_show:
        return

    if args.backend in {"auto", "open3d"}:
        try:
            show_open3d(shown_points, colors, args.point_size)
            return
        except Exception as exc:
            if args.backend == "open3d":
                raise
            print("[WARN] Open3D unavailable, using matplotlib instead: {}".format(exc))

    matplotlib_preview(shown_points, colors, args.lidar_path, output_path=None, show=True)


if __name__ == "__main__":
    main()
