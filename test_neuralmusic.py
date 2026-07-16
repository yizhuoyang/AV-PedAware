import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader

from dataloader.NeuralMusic_loader import AVPedNeuralMusicLoader
from network.NeuralMUSIC import NeuralMusic


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NeuralMUSIC DOA on paired ROS1 data")
    parser.add_argument("--data-root", default="data/pairs_ros1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--object-filter", default="person")
    parser.add_argument("--audio-channels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--feature-type", choices=["magphase", "ipd"], default="ipd")
    parser.add_argument("--checkpoint", default="output_neuralmusic_ros1_person/model_best.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--axis-limit", type=float, default=5.3)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples evaluated in total. Zero evaluates the full split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pairs_ros1/neuralmusic_eval"),
    )
    parser.add_argument(
        "--max-vis",
        type=int,
        default=0,
        help="Maximum visualizations saved per sequence. Zero exports every frame.",
    )
    return parser.parse_args()


def circular_error_deg(prediction_deg, target_deg):
    difference = np.deg2rad(prediction_deg - target_deg)
    return np.degrees(np.abs(np.arctan2(np.sin(difference), np.cos(difference))))


def read_box(path):
    fields = Path(path).read_text().split()
    if len(fields) != 15:
        return None
    h, w, length = map(float, fields[8:11])
    x, y, z, yaw = map(float, fields[11:15])
    return np.asarray([x, y, z, length, w, h, yaw], dtype=np.float32)


def bbox_bev_corners(box):
    x, y, _, length, width, _, yaw = box
    corners = np.asarray(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
    )
    rotation = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    return corners @ rotation.T + np.asarray([x, y])


def draw_ray(ax, angle_deg, color, label, length=4.7):
    angle = np.deg2rad(angle_deg)
    ax.plot(
        [0.0, length * np.cos(angle)],
        [0.0, length * np.sin(angle)],
        color=color,
        linewidth=2.1,
        label=label,
    )


def render_visualization(row, axis_limit):
    points = np.fromfile(row["lidar"], dtype=np.float32).reshape(-1, 4)
    box = read_box(row["label"])
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.02, 1.0]}
    )
    scatter = axes[0].scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        s=0.35,
        cmap="viridis",
        linewidths=0,
    )
    if box is not None:
        axes[0].add_patch(
            Polygon(
                bbox_bev_corners(box),
                closed=True,
                fill=False,
                edgecolor="red",
                linewidth=2.0,
                label="bbox",
            )
        )
    draw_ray(axes[0], row["gt_deg"], "cyan", "GT DOA")
    draw_ray(axes[0], row["pred_deg"], "orange", "NeuralMUSIC")
    axes[0].scatter([0], [0], color="white", edgecolor="black", s=35, zorder=5)
    axes[0].set_xlim(-axis_limit, axis_limit)
    axes[0].set_ylim(-axis_limit, axis_limit)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title(
        "{} / {}\nGT {:.1f} deg | Pred {:.1f} deg | Err {:.1f} deg".format(
            row["sequence"], row["stem"], row["gt_deg"], row["pred_deg"], row["error_deg"]
        )
    )
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=axes[0], fraction=0.045, pad=0.03, label="z (m)")

    angles = np.arange(len(row["spectrum"]), dtype=np.float32)
    spectrum = np.asarray(row["spectrum"])
    spectrum = spectrum / max(float(spectrum.max()), 1e-8)
    axes[1].fill_between(angles, spectrum, color="#2a9d8f", alpha=0.2)
    axes[1].plot(angles, spectrum, color="#176b63", linewidth=1.5, label="Spectrum")
    axes[1].axvline(row["gt_deg"], color="cyan", linewidth=2.0, label="GT DOA")
    axes[1].axvline(row["pred_deg"], color="orange", linewidth=2.0, label="Argmax")
    axes[1].set_xlim(0, 360)
    axes[1].set_xticks(np.arange(0, 361, 60))
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Azimuth (deg)")
    axes[1].set_ylabel("Normalized spectrum")
    axes[1].set_title("NeuralMUSIC output spectrum")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = AVPedNeuralMusicLoader(
        root_path=args.data_root,
        split=args.split,
        audio_channels=tuple(args.audio_channels),
        feature_type=args.feature_type,
        object_filter=args.object_filter,
        geometry_aug=False,
        return_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    input_channel = (
        1 + 2 * (len(args.audio_channels) - 1)
        if args.feature_type == "ipd"
        else 2 * len(args.audio_channels)
    )
    model = NeuralMusic(
        N=len(args.audio_channels),
        T=8000,
        M=1,
        device=device,
        attention=not args.no_attention,
        input_channel=input_channel,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    records = []
    with torch.no_grad():
        for spec, gt_deg, steering, correlation, metadata in loader:
            _, spectrum = model(
                spec.to(device).float(),
                steering.to(device),
                correlation.to(device),
            )
            spectrum = spectrum.cpu().numpy()
            prediction = np.argmax(spectrum, axis=1).astype(np.float32)
            for index in range(len(prediction)):
                if args.max_samples > 0 and len(records) >= args.max_samples:
                    break
                target = float(gt_deg[index, 0])
                records.append(
                    {
                        "sequence": metadata["sequence"][index],
                        "stem": metadata["stem"][index],
                        "gt_deg": target,
                        "pred_deg": float(prediction[index]),
                        "error_deg": float(circular_error_deg(prediction[index], target)),
                        "confidence": float(spectrum[index].max()),
                        "label": metadata["label"][index],
                        "lidar": metadata["lidar"][index],
                        "spectrum": spectrum[index],
                    }
                )
            if args.max_samples > 0 and len(records) >= args.max_samples:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "spectrum"} for row in records]
    ).sort_values(["sequence", "stem"])
    results_path = args.output_dir / "results.csv"
    table.to_csv(results_path, index=False)
    summary = table.groupby("sequence").agg(
        frames=("stem", "count"),
        mae_deg=("error_deg", "mean"),
        median_deg=("error_deg", "median"),
        max_deg=("error_deg", "max"),
    )
    summary.loc["ALL"] = [
        len(table),
        table["error_deg"].mean(),
        table["error_deg"].median(),
        table["error_deg"].max(),
    ]
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path)

    record_lookup = {(row["sequence"], row["stem"]): row for row in records}
    saved = 0
    for sequence, group in table.groupby("sequence"):
        rows = group if args.max_vis == 0 else group.head(args.max_vis)
        vis_dir = args.output_dir / "visualizations" / sequence
        vis_dir.mkdir(parents=True, exist_ok=True)
        for _, row in rows.iterrows():
            record = record_lookup[(row["sequence"], row["stem"])]
            figure = render_visualization(record, args.axis_limit)
            figure.savefig(vis_dir / "{}.png".format(row["stem"]), dpi=130, bbox_inches="tight")
            plt.close(figure)
            saved += 1

    print("sequences: {}".format(dataset.selected_sequences))
    print(summary.round(3).to_string())
    print("results: {}".format(results_path))
    print("summary: {}".format(summary_path))
    print("visualizations: {} files under {}".format(saved, args.output_dir / "visualizations"))


if __name__ == "__main__":
    main(parse_args())
