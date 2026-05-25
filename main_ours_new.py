import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloader.NeuralMusic_loader import AVPedNeuralMusicLoader
from dataloader.AudioNavNeuralMusic_loader import AudioNavNeuralMusicLoader
from network.NeuralMUSIC import NeuralMusic
from utils.model_training import ModelTrainer


def parse_mic_offsets(value):
    if not value:
        return None
    rows = []
    for row in value.split(";"):
        rows.append([float(v) for v in row.split(",")])
    offsets = np.asarray(rows, dtype=np.float32)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("mic offsets must look like 'x,y,z;x,y,z;...'")
    return offsets


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mic_offsets = parse_mic_offsets(args.mic_offsets)
    uses_audio_nav_layout = args.dataset_type in {"audio-nav", "respeaker"}
    dataset_cls = AudioNavNeuralMusicLoader if uses_audio_nav_layout else AVPedNeuralMusicLoader
    audio_nav_kwargs = (
        {
            "doa_field": args.doa_field,
            "train_ratio": args.train_ratio,
            "object_filter": args.object_filter,
            "test_sequences": parse_csv_list(args.test_sequences),
        }
        if uses_audio_nav_layout
        else {}
    )

    train_dataset = dataset_cls(
        root_path=args.data_root,
        split=args.train_split,
        mic_offsets=mic_offsets,
        audio_channels=tuple(args.audio_channels),
        feature_type=args.feature_type,
        geometry_aug=args.geometry_aug,
        rotation_interval=args.rotation_interval,
        **audio_nav_kwargs,
    )
    val_dataset = dataset_cls(
        root_path=args.data_root,
        split=args.val_split,
        mic_offsets=mic_offsets,
        audio_channels=tuple(args.audio_channels),
        feature_type=args.feature_type,
        geometry_aug=False,
        **audio_nav_kwargs,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and validation splits must both contain samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_dataset,
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
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"dataset type: {args.dataset_type}")
    if uses_audio_nav_layout:
        print(f"object filter: {args.object_filter or 'all'}")
        print(f"doa field: {args.doa_field}")
        print(f"test sequences override: {args.test_sequences or 'none'}")
        print(f"train sequences: {train_dataset.selected_sequences}")
        print(f"val sequences: {val_dataset.selected_sequences}")
    print(f"audio channels: {args.audio_channels}")
    print(f"feature type: {args.feature_type}")
    print(f"geometry augmentation: {args.geometry_aug}")
    print(f"input feature channels: {input_channel}")
    print(f"device: {device}")

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=None,
        optimizer=optimizer,
        epoch=args.epochs,
        model_path=str(save_dir),
        device=device,
        lr_scheduler=scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuralMUSIC on AV-PedAware pairs data")
    parser.add_argument("--dataset-type", choices=["avped", "audio-nav", "respeaker"], default="avped")
    parser.add_argument("--data-root", default="data/pairs")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--doa-field",
        default="auto",
        help=(
            "DOA target field. auto uses heading_target_yaw_signed_deg for doa_lio_odom.npz "
            "and azimuth_rad for per-sample doa/*.npy."
        ),
    )
    parser.add_argument("--object-filter", default="", help="Comma-separated sequence prefixes, e.g. clock or clock,dryer")
    parser.add_argument("--test-sequences", default="", help="Comma-separated sequence names reserved for val/test, e.g. clock2")
    parser.add_argument("--audio-channels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--feature-type", choices=["magphase", "ipd"], default="magphase")
    parser.add_argument("--geometry-aug", action="store_true")
    parser.add_argument("--rotation-interval", type=int, default=None)
    parser.add_argument("--mic-offsets", default="", help="Optional 'x,y,z;x,y,z;...' microphone geometry in meters")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--save-dir", default="output_neuralmusic")
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())

