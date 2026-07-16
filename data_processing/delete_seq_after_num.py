#!/usr/bin/env python3
"""Delete sequence modality files whose numeric sample id is greater than a limit."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-dir",
        type=Path,
        required=True,
        help="Sequence directory containing modality subdirectories such as audio/lidar/labels.",
    )
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="Keep samples with numeric stem <= num and delete files after it.",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=None,
        help="Optional modality subdirectory names to process. Default processes all subdirectories.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only print a preview.",
    )
    return parser.parse_args()


def numeric_stem(path):
    try:
        return int(path.stem)
    except ValueError:
        return None


def iter_modality_dirs(seq_dir, modality_names):
    if modality_names:
        for name in modality_names:
            modality_dir = seq_dir / name
            if modality_dir.is_dir():
                yield modality_dir
            else:
                print("skip missing modality: {}".format(modality_dir))
        return

    yield from sorted(path for path in seq_dir.iterdir() if path.is_dir())


def main():
    args = parse_args()
    if args.num < 0:
        raise ValueError("--num must be non-negative")
    if not args.seq_dir.is_dir():
        raise FileNotFoundError("Missing sequence directory: {}".format(args.seq_dir))

    files_to_delete = []
    skipped_non_numeric = 0
    modality_counts = {}

    for modality_dir in iter_modality_dirs(args.seq_dir, args.modalities):
        selected = []
        for path in sorted(item for item in modality_dir.iterdir() if item.is_file()):
            sample_id = numeric_stem(path)
            if sample_id is None:
                skipped_non_numeric += 1
                continue
            if sample_id > args.num:
                selected.append(path)
        if selected:
            modality_counts[modality_dir.name] = len(selected)
            files_to_delete.extend(selected)

    print("seq_dir: {}".format(args.seq_dir))
    print("keep numeric sample id <= {}".format(args.num))
    print("matched files after num: {}".format(len(files_to_delete)))
    for modality, count in sorted(modality_counts.items()):
        print("{}: {}".format(modality, count))
    if skipped_non_numeric:
        print("skipped non-numeric file stems: {}".format(skipped_non_numeric))

    if args.apply:
        for path in files_to_delete:
            path.unlink()
        print("deleted files: {}".format(len(files_to_delete)))
    else:
        preview = files_to_delete[:20]
        for path in preview:
            print("would delete {}".format(path))
        if len(files_to_delete) > len(preview):
            print("... {} more".format(len(files_to_delete) - len(preview)))
        print("Preview only. Re-run with --apply to delete these files.")


if __name__ == "__main__":
    main()
