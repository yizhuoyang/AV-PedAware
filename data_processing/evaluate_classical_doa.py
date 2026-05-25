#!/usr/bin/env python3
"""Evaluate a classical SRP-PHAT DOA baseline on paired audio/lidar samples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy import signal
from scipy.io import wavfile


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-root", type=Path, default=Path("data/pairs_ros1"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--object-filter",
        default="person",
        help="Comma-separated sequence prefixes. Empty string processes all sequences.",
    )
    parser.add_argument("--sequences", nargs="*", help="Optional exact sequence names.")
    parser.add_argument("--audio-channels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--mic-spacing-mm", type=float, default=45.7)
    parser.add_argument("--speed-of-sound", type=float, default=343.0)
    parser.add_argument("--angle-bins", type=int, default=360)
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument(
        "--steering-sign",
        type=int,
        choices=[-1, 1],
        default=1,
        help="Flip this value if the spectrum is mirrored by channel/array orientation.",
    )
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--low-cut-hz", type=float, default=300.0)
    parser.add_argument("--high-cut-hz", type=float, default=3000.0)
    parser.add_argument("--axis-limit", type=float, default=5.3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pairs_ros1/classical_srp_phat_eval"),
    )
    parser.add_argument(
        "--max-vis",
        type=int,
        default=0,
        help="Maximum saved visualizations per sequence. Zero exports every sample.",
    )
    return parser.parse_args()


def microphone_positions(spacing_mm):
    side = spacing_mm / 1000.0 / 2.0
    return np.asarray(
        [
            [side, side],
            [-side, side],
            [-side, -side],
            [side, -side],
        ],
        dtype=np.float32,
    )


def iter_sequences(args):
    split_dir = args.pairs_root / args.split
    if not split_dir.is_dir():
        raise FileNotFoundError("Dataset split directory does not exist: {}".format(split_dir))
    selected = set(args.sequences or [])
    prefixes = tuple(name.strip() for name in args.object_filter.split(",") if name.strip())
    for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        if selected and sequence_dir.name not in selected:
            continue
        if prefixes and not sequence_dir.name.startswith(prefixes):
            continue
        if (sequence_dir / "audio").is_dir() and (sequence_dir / "labels").is_dir():
            yield sequence_dir


def read_label_doa(label_path):
    lines = [line for line in label_path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Expected one bbox in {}, got {}".format(label_path, len(lines)))
    fields = lines[0].split()
    if len(fields) != 15:
        raise ValueError("Expected 15 fields in {}, got {}".format(label_path, len(fields)))
    x, y = map(float, fields[11:13])
    return float(np.degrees(np.arctan2(y, x)) % 360.0)


def read_label_box(label_path):
    fields = label_path.read_text().split()
    if len(fields) != 15:
        return None
    h, w, length = map(float, fields[8:11])
    x, y, z, yaw = map(float, fields[11:15])
    return np.asarray([x, y, z, length, w, h, yaw], dtype=np.float32)


def circular_error_deg(prediction_deg, target_deg):
    difference = np.deg2rad(prediction_deg - target_deg)
    return float(np.degrees(np.abs(np.arctan2(np.sin(difference), np.cos(difference)))))


def srp_phat_spectrum(audio_path, args, mic_positions):
    sample_rate, audio = wavfile.read(audio_path)
    if audio.ndim == 1:
        audio = audio[:, None]
    if max(args.audio_channels) >= audio.shape[1]:
        raise ValueError(
            "Requested channels {} but {} has shape {}".format(
                args.audio_channels, audio_path, audio.shape
            )
        )
    if len(args.audio_channels) != len(mic_positions):
        raise ValueError("SRP-PHAT geometry requires four selected audio channels.")
    wave = audio[:, args.audio_channels].astype(np.float32)
    wave -= wave.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(wave))
    if scale > 0.0:
        wave /= scale

    frequencies, _, stft = signal.stft(
        wave.T,
        fs=sample_rate,
        window="hann",
        nperseg=args.n_fft,
        noverlap=args.n_fft - args.hop_length,
        nfft=args.n_fft,
        boundary=None,
        padded=False,
        axis=-1,
    )
    frequency_mask = (frequencies >= args.low_cut_hz) & (frequencies <= args.high_cut_hz)
    frequencies = frequencies[frequency_mask]
    stft = stft[:, frequency_mask, :]

    world_angles_deg = np.arange(args.angle_bins, dtype=np.float32) * (
        360.0 / args.angle_bins
    )
    microphone_angles = np.deg2rad(world_angles_deg - args.angle_offset_deg)
    directions = np.stack([np.cos(microphone_angles), np.sin(microphone_angles)], axis=1)

    scores = np.zeros(args.angle_bins, dtype=np.float64)
    for first in range(len(mic_positions)):
        for second in range(first + 1, len(mic_positions)):
            cross = stft[first] * np.conj(stft[second])
            cross /= np.maximum(np.abs(cross), 1e-8)
            cross = cross.mean(axis=1)
            delay = ((mic_positions[first] - mic_positions[second]) @ directions.T) / args.speed_of_sound
            phase = args.steering_sign * 2.0 * np.pi * frequencies[:, None] * delay[None, :]
            scores += np.real(cross[:, None] * np.exp(-1j * phase)).sum(axis=0)

    spectrum = scores - scores.min()
    if spectrum.sum() > 0.0:
        spectrum /= spectrum.sum()
    predicted_bin = int(np.argmax(spectrum))
    return world_angles_deg, spectrum.astype(np.float32), float(world_angles_deg[predicted_bin])


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


def render_visualization(row, args):
    points = np.fromfile(row["lidar"], dtype=np.float32).reshape(-1, 4)
    box = read_label_box(Path(row["label"]))
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
    draw_ray(axes[0], row["pred_deg"], "orange", "SRP-PHAT")
    axes[0].scatter([0], [0], color="white", edgecolor="black", s=35, zorder=5)
    axes[0].set_xlim(-args.axis_limit, args.axis_limit)
    axes[0].set_ylim(-args.axis_limit, args.axis_limit)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title(
        "{} / {}\nGT {:.1f} deg | SRP-PHAT {:.1f} deg | Err {:.1f} deg".format(
            row["sequence"], row["stem"], row["gt_deg"], row["pred_deg"], row["error_deg"]
        )
    )
    axes[0].legend(loc="upper right", fontsize=8)
    fig.colorbar(scatter, ax=axes[0], fraction=0.045, pad=0.03, label="z (m)")

    angles = np.asarray(row["angle_grid_deg"])
    spectrum = np.asarray(row["spectrum"])
    axes[1].fill_between(angles, spectrum, color="#2a9d8f", alpha=0.2)
    axes[1].plot(angles, spectrum, color="#176b63", linewidth=1.5, label="SRP-PHAT")
    axes[1].axvline(row["gt_deg"], color="cyan", linewidth=2.0, label="GT DOA")
    axes[1].axvline(row["pred_deg"], color="orange", linewidth=2.0, label="Peak")
    axes[1].set_xlim(0, 360)
    axes[1].set_xticks(np.arange(0, 361, 60))
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Azimuth (deg)")
    axes[1].set_ylabel("Normalized response")
    axes[1].set_title("Classical SRP-PHAT spatial spectrum")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def evaluate(args):
    mic_positions = microphone_positions(args.mic_spacing_mm)
    records = []
    for sequence_dir in iter_sequences(args):
        audio_paths = sorted((sequence_dir / "audio").glob("*.wav"))
        for audio_path in audio_paths:
            stem = audio_path.stem
            label_path = sequence_dir / "labels" / "{}.txt".format(stem)
            lidar_path = sequence_dir / "lidar" / "{}.bin".format(stem)
            if not label_path.exists() or not lidar_path.exists() or not label_path.read_text().strip():
                continue
            gt_deg = read_label_doa(label_path)
            angle_grid, spectrum, pred_deg = srp_phat_spectrum(audio_path, args, mic_positions)
            records.append(
                {
                    "sequence": sequence_dir.name,
                    "stem": stem,
                    "gt_deg": gt_deg,
                    "pred_deg": pred_deg,
                    "error_deg": circular_error_deg(pred_deg, gt_deg),
                    "peak_response": float(spectrum.max()),
                    "audio": str(audio_path),
                    "label": str(label_path),
                    "lidar": str(lidar_path),
                    "angle_grid_deg": angle_grid,
                    "spectrum": spectrum,
                }
            )
    if not records:
        raise ValueError("No valid samples found for classical DOA evaluation.")
    return records


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = evaluate(args)
    table = pd.DataFrame(
        [{key: value for key, value in row.items() if key not in {"angle_grid_deg", "spectrum"}} for row in records]
    )
    table = table.sort_values(["sequence", "stem"]).reset_index(drop=True)
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

    visualizations = []
    for sequence, indices in table.groupby("sequence").groups.items():
        selected_indices = list(indices)
        if args.max_vis > 0:
            selected_indices = selected_indices[: args.max_vis]
        sequence_dir = args.output_dir / "visualizations" / sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)
        for index in selected_indices:
            output_path = sequence_dir / "{}.png".format(table.loc[index, "stem"])
            fig = render_visualization(records[index], args)
            fig.savefig(output_path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            visualizations.append(str(output_path))

    print("method: SRP-PHAT")
    print("mic spacing: {:.1f} mm channels: {}".format(args.mic_spacing_mm, args.audio_channels))
    print("angle offset: {:.1f} deg steering sign: {}".format(args.angle_offset_deg, args.steering_sign))
    print(summary.round(3).to_string())
    print("results: {}".format(results_path))
    print("summary: {}".format(summary_path))
    print("visualizations: {} files under {}".format(len(visualizations), args.output_dir / "visualizations"))


if __name__ == "__main__":
    main()
