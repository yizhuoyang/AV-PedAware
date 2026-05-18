import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.avped_pairs_dataloader import AVpedPairsLoader
from network.avped import FusionNet
from utils.loss import regression_loss


def run_epoch(model, dataloader, optimizer, device, training):
    model.train(training)
    total_loss = 0.0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for spec, image, _, gt in tqdm(dataloader, leave=False):
            spec = spec.to(device)
            image = image.to(device)
            gt = gt.to(device)

            if training:
                optimizer.zero_grad()

            pred, _, _ = model(spec, image)
            loss = regression_loss(pred, gt)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def build_loader(args, split, shuffle):
    dataset = AVpedPairsLoader(
        root_path=args.data_root,
        split=split,
        audio_channels=tuple(args.audio_channels),
        include_lidar=False,
        return_metadata=False,
        skip_empty_labels=True,
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
    if len(train_dataset) == 0:
        raise ValueError(f"No samples found for train split: {args.train_split}")
    if len(val_dataset) == 0:
        raise ValueError(f"No samples found for val split: {args.val_split}")

    model = FusionNet(
        dropout_rate=args.dropout_rate,
        audio_channels=len(args.audio_channels),
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    best_val_loss = float("inf")

    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"audio channels: {args.audio_channels}")
    print(f"device: {device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, training=False)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        torch.save(model.state_dict(), save_dir / "last.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "model_best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FusionNet on the new paired dataset format")
    parser.add_argument("--data-root", default="data/pairs")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--audio-channels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.6)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-dir", default="output_pairs")
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
