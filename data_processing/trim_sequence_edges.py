#!/usr/bin/env python3
"""Remove matching samples from the beginning and end of each paired sequence."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-root",
        type=Path,
        default=Path("data/pairs_ros1"),
        help="Dataset root containing split/sequence/modality directories.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to process below pairs-root.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        help="Optional sequence names to process. Default processes every sequence.",
    )
    parser.add_argument(
        "--edge-count",
        type=int,
        default=10,
        help="Number of samples to remove at both the beginning and end.",
    )
    parser.add_argument(
        "--anchor-modality",
        default="lidar",
        help="Modality used to determine ordered sample stems.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only print a preview.",
    )
    return parser.parse_args()


def iter_sequences(args):
    selected = set(args.sequences) if args.sequences else None
    for split in args.splits:
        split_dir = args.pairs_root / split
        if not split_dir.exists():
            print("skip missing split: {}".format(split_dir))
            continue
        for sequence_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            if selected is None or sequence_dir.name in selected:
                yield split, sequence_dir


def ordered_anchor_stems(sequence_dir, anchor_modality):
    anchor_dir = sequence_dir / anchor_modality
    if not anchor_dir.is_dir():
        return []
    return sorted({path.stem for path in anchor_dir.iterdir() if path.is_file()})


def files_for_stems(sequence_dir, stems):
    matched = []
    for modality_dir in sorted(path for path in sequence_dir.iterdir() if path.is_dir()):
        for path in sorted(item for item in modality_dir.iterdir() if item.is_file()):
            if path.stem in stems:
                matched.append(path)
    return matched


def main():
    args = parse_args()
    if args.edge_count < 0:
        raise ValueError("--edge-count must be non-negative")
    if args.edge_count == 0:
        print("edge_count is 0; nothing to remove.")
        return

    sequences_processed = 0
    samples_selected = 0
    files_selected = 0

    for split, sequence_dir in iter_sequences(args):
        stems = ordered_anchor_stems(sequence_dir, args.anchor_modality)
        if len(stems) <= 2 * args.edge_count:
            print(
                "skip {}/{}: {} anchor samples are insufficient for trimming {} from each edge".format(
                    split, sequence_dir.name, len(stems), args.edge_count
                )
            )
            continue

        remove_stems = set(stems[: args.edge_count] + stems[-args.edge_count :])
        files = files_for_stems(sequence_dir, remove_stems)
        sequences_processed += 1
        samples_selected += len(remove_stems)
        files_selected += len(files)
        print(
            "{}/{}: samples={} files={} first={} last={}".format(
                split,
                sequence_dir.name,
                len(remove_stems),
                len(files),
                stems[: args.edge_count],
                stems[-args.edge_count :],
            )
        )

        if args.apply:
            for path in files:
                path.unlink()

    action = "deleted" if args.apply else "would delete"
    print("{} sequences: {}".format(action, sequences_processed))
    print("{} paired samples: {}".format(action, samples_selected))
    print("{} modality files: {}".format(action, files_selected))
    if not args.apply:
        print("Preview only. Re-run with --apply to delete these files.")


if __name__ == "__main__":
    main()
