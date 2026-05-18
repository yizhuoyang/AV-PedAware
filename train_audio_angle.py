import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.avped_audio_angle_dataloader import AVpedAudioAngleLoader
from network.audio_angle_net import AudioAngleNet
from utils.loss import angle_vector_loss


def angular_error_deg(pred_vector, target_vector):
    pred_angle = torch.atan2(pred_vector[:, 0], pred_vector[:, 1])
    target_angle = torch.atan2(target_vector[:, 0], target_vector[:, 1])
    diff = torch.atan2(torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle))
    return torch.rad2deg(torch.abs(diff))


def run_epoch(model, dataloader, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    errors = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for spec, target in tqdm(dataloader, leave=False):
            spec = spec.to(device)
            target = target.to(device)
            if training:
                optimizer.zero_grad()
            pred = model(spec)
            loss = angle_vector_loss(pred, target)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            errors.append(angular_error_deg(pred, target).detach().cpu().numpy())
    return total_loss / max(len(dataloader), 1), float(np.concatenate(errors).mean())


def build_loader(args, split, shuffle):
    dataset = AVpedAudioAngleLoader(
        root_path=args.data_root,
        split=split,
        audio_channels=tuple(args.audio_channels),
        feature_type=args.feature_type,
        augment=shuffle,
        freq_mask_param=args.freq_mask_param,
        time_mask_param=args.time_mask_param,
    )
    return dataset, DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        drop_last=shuffle,
    )


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader = build_loader(args, args.train_split, shuffle=True)
    val_dataset, val_loader = build_loader(args, args.val_split, shuffle=False)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Train and validation splits must both contain samples")

    model = AudioAngleNet(
        dropout_rate=args.dropout_rate,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        kernel_num=args.kernel_num,
        audio_channels=1 + 2 * (len(args.audio_channels) - 1) if args.feature_type == "ipd" else len(args.audio_channels),
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    best_val_error = float("inf")
    epochs_without_improvement = 0

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"audio channels: {args.audio_channels}")
    print(f"feature type: {args.feature_type}")
    print(f"device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss, val_mae = run_epoch(model, val_loader, optimizer, device, training=False)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} train_mae_deg={train_mae:.3f} "
            f"val_loss={val_loss:.6f} val_mae_deg={val_mae:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step(val_mae)
        torch.save(model.state_dict(), save_dir / "last.pth")
        if val_mae < best_val_error:
            best_val_error = val_mae
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_dir / "model_best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}; best_val_mae_deg={best_val_error:.3f}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train audio-only azimuth estimator")
    parser.add_argument("--data-root", default="data/pairs")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--audio-channels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--feature-type", choices=["ipd", "mel"], default="ipd")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout-rate", type=float, default=0.4)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--kernel-num", type=int, default=8)
    parser.add_argument("--freq-mask-param", type=int, default=8)
    parser.add_argument("--time-mask-param", type=int, default=8)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-dir", default="output_audio_angle")
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
