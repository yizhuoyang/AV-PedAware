import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.avped_pairs_dataloader import AVpedPairsLoader
from network.avped import FusionNet
from utils.loss import regression_loss


def bbox_bev_corners(box):
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
    return corners @ rotation.T + np.array([x, y], dtype=np.float32)


def build_dataset(args, include_lidar=False, return_metadata=False):
    return AVpedPairsLoader(
        root_path=args.data_root,
        split=args.split,
        audio_channels=tuple(args.audio_channels),
        include_lidar=include_lidar,
        return_metadata=return_metadata,
        skip_empty_labels=True,
    )


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    abs_xyz = []
    with torch.no_grad():
        for spec, image, _, gt in tqdm(dataloader, leave=False):
            spec = spec.to(device)
            image = image.to(device)
            gt = gt.to(device)
            pred, _, _ = model(spec, image)
            total_loss += regression_loss(pred, gt).item()
            abs_xyz.append(torch.abs(pred[:, :3] - gt[:, :3]).cpu().numpy())

    abs_xyz = np.concatenate(abs_xyz, axis=0)
    return {
        "loss": total_loss / max(len(dataloader), 1),
        "mae_x": float(abs_xyz[:, 0].mean()),
        "mae_y": float(abs_xyz[:, 1].mean()),
        "mae_z": float(abs_xyz[:, 2].mean()),
    }


def visualize_sample(model, dataset, index, device, output_path=None):
    spec, image, depth, gt, lidar, meta = dataset[index]
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(
            spec.unsqueeze(0).to(device),
            image.unsqueeze(0).to(device),
        )
    pred = pred[0].cpu().numpy()
    gt = gt.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].imshow(image.permute(1, 2, 0).numpy() * 0.5 + 0.5)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    depth_plot = axes[1].imshow(depth[0].numpy(), cmap="viridis")
    axes[1].set_title("Depth image")
    axes[1].axis("off")
    fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(spec.numpy()[0], origin="lower", aspect="auto", cmap="magma")
    axes[2].set_title("Audio mel spectrogram, channel 0")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Mel bins")

    scatter = axes[3].scatter(lidar[:, 0], lidar[:, 1], c=lidar[:, 2], s=0.35, cmap="viridis")
    axes[3].add_patch(
        Polygon(
            bbox_bev_corners(gt),
            closed=True,
            fill=False,
            edgecolor="red",
            linewidth=2,
            label="GT",
        )
    )
    axes[3].add_patch(
        Polygon(
            bbox_bev_corners(pred),
            closed=True,
            fill=False,
            edgecolor="lime",
            linewidth=2,
            label="Prediction",
        )
    )
    axes[3].set_title(f"LiDAR BEV + bbox\n{meta['bag']} / {meta['stem']}")
    axes[3].set_xlabel("x (m)")
    axes[3].set_ylabel("y (m)")
    axes[3].set_xlim(-4.1, 4.1)
    axes[3].set_ylim(-4.1, 4.1)
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].legend(loc="upper right")
    fig.colorbar(scatter, ax=axes[3], fraction=0.046, pad=0.04, label="z (m)")

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    else:
        plt.show()
    plt.close(fig)

    print("meta:", meta)
    print("gt:", gt.tolist())
    print("pred:", pred.tolist())


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = FusionNet(audio_channels=len(args.audio_channels)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    eval_dataset = build_dataset(args)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    metrics = evaluate(model, eval_loader, device)
    print(
        "loss={loss:.6f} mae_x={mae_x:.4f} mae_y={mae_y:.4f} mae_z={mae_z:.4f}".format(
            **metrics
        )
    )

    vis_dataset = build_dataset(args, include_lidar=True, return_metadata=True)
    index = args.index if args.index >= 0 else random.randrange(len(vis_dataset))
    visualize_sample(model, vis_dataset, index, device, args.save_figure or None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FusionNet on the new paired dataset format")
    parser.add_argument("--data-root", default="data/pairs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-channels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--checkpoint", default="output_pairs/model_best.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index", type=int, default=-1, help="Sample index to visualize; -1 selects randomly")
    parser.add_argument("--save-figure", default="", help="Optional path to save the visualization figure")
    main(parser.parse_args())
