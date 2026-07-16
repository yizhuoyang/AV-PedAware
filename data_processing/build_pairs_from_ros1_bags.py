#!/usr/bin/env python3
"""Build timestamp-aligned audio/image/depth/lidar samples from ROS 1 bags.

Wav names must include the corresponding bag stem followed by the audio start
timestamp, for example:

    my_recording_1747800000.125.wav  ->  my_recording.bag

The timestamp is interpreted as Unix/ROS time and may be supplied in seconds
or integer nanoseconds.
"""

import argparse
import csv
from bisect import bisect_left
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile


DEFAULT_COLOR_TOPIC = "/camera/color/image_raw"
DEFAULT_DEPTH_TOPIC = "/camera/depth/image_rect_raw"
DEFAULT_LIDAR_TOPIC = "/livox/lidar"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", type=Path, required=True, help="Directory containing timestamped wav files.")
    parser.add_argument("--bag-dir", type=Path, required=True, help="Directory containing corresponding ROS 1 .bag files.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/pairs_ros1"),
        help="Root directory for paired samples.",
    )
    parser.add_argument("--segment-seconds", type=float, default=0.5, help="Audio segment duration in seconds.")
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="Audio sliding-window hop in seconds. Defaults to segment-seconds for non-overlapping segments.",
    )
    parser.add_argument(
        "--max-sync-gap-seconds",
        type=float,
        default=0.25,
        help="Maximum allowed gap between the audio segment center and a sensor frame.",
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help="Unit of the final timestamp token in each wav stem.",
    )
    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument(
        "--header-stamp",
        action="store_true",
        help="Synchronize using each message header stamp instead of rosbag recorded time.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search wav and bag directories recursively.")
    parser.add_argument("--bag-name", default="", help="Only process one bag stem.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in an existing output bag directory.")
    return parser.parse_args()


def parse_start_stamp_ns(text, unit):
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("invalid timestamp: {}".format(text)) from exc
    if unit == "auto":
        absolute = abs(value)
        if absolute >= Decimal("1e17"):
            unit = "ns"
        elif absolute >= Decimal("1e14"):
            unit = "us"
        elif absolute >= Decimal("1e11"):
            unit = "ms"
        else:
            unit = "s"
    scale = {
        "s": Decimal("1e9"),
        "ms": Decimal("1e6"),
        "us": Decimal("1e3"),
        "ns": Decimal("1"),
    }[unit]
    return int((value * scale).to_integral_value(rounding=ROUND_HALF_UP))


def find_input_files(directory, pattern, recursive):
    return sorted(directory.rglob(pattern) if recursive else directory.glob(pattern))


def match_wavs_to_bags(wav_dir, bag_dir, timestamp_unit, recursive, bag_name):
    bag_paths = find_input_files(bag_dir, "*.bag", recursive)
    if bag_name:
        bag_paths = [path for path in bag_paths if path.stem == bag_name]
    bag_paths.sort(key=lambda path: len(path.stem), reverse=True)

    jobs = []
    for wav_path in find_input_files(wav_dir, "*.wav", recursive):
        bag_path = next(
            (path for path in bag_paths if wav_path.stem.startswith(path.stem + "_")),
            None,
        )
        if bag_path is None:
            print("[SKIP] {}: no matching .bag stem found".format(wav_path))
            continue
        timestamp_text = wav_path.stem[len(bag_path.stem) + 1 :]
        try:
            start_ns = parse_start_stamp_ns(timestamp_text, timestamp_unit)
        except ValueError:
            print("[SKIP] {}: invalid start timestamp '{}'".format(wav_path, timestamp_text))
            continue
        jobs.append((bag_path, wav_path, start_ns))
    return jobs


def message_stamp_ns(msg, recorded_time, use_header_stamp):
    if use_header_stamp and hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp
        if hasattr(stamp, "to_nsec"):
            value = stamp.to_nsec()
            if value:
                return value
    return recorded_time.to_nsec()


def load_sensor_messages(bag_path, topics, use_header_stamp):
    try:
        import rosbag
    except ImportError as exc:
        raise ImportError(
            "The ROS 1 Python package 'rosbag' is required. "
            "Source your ROS 1 environment before running this script."
        ) from exc

    messages = {topic: [] for topic in topics}
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, recorded_time in bag.read_messages(topics=list(topics)):
            messages[topic].append(
                (message_stamp_ns(msg, recorded_time, use_header_stamp), msg)
            )
    missing = [topic for topic, rows in messages.items() if not rows]
    if missing:
        raise ValueError("Missing required topics: {}".format(", ".join(missing)))
    for rows in messages.values():
        rows.sort(key=lambda row: row[0])
    return messages


def nearest_message(rows, target_ns, max_gap_ns):
    timestamps = [row[0] for row in rows]
    idx = bisect_left(timestamps, target_ns)
    candidates = []
    if idx < len(rows):
        candidates.append(rows[idx])
    if idx > 0:
        candidates.append(rows[idx - 1])
    if not candidates:
        return None
    selected = min(candidates, key=lambda row: abs(row[0] - target_ns))
    return selected if abs(selected[0] - target_ns) <= max_gap_ns else None


def decode_compressed_image(msg, is_depth):
    payload = bytes(msg.data)
    if is_depth and "compressedDepth" in getattr(msg, "format", ""):
        signature = b"\x89PNG\r\n\x1a\n"
        start = payload.find(signature)
        if start < 0:
            raise ValueError("Unable to find PNG payload in compressedDepth message")
        payload = payload[start:]
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("Failed to decode compressed {} image".format("depth" if is_depth else "color"))
    return image


def decode_raw_image(msg, is_depth):
    encoding = str(msg.encoding).lower()
    dtype_by_encoding = {
        "mono8": np.uint8,
        "8uc1": np.uint8,
        "bgr8": np.uint8,
        "rgb8": np.uint8,
        "mono16": np.uint16,
        "16uc1": np.uint16,
        "32fc1": np.float32,
    }
    if encoding not in dtype_by_encoding:
        raise ValueError("Unsupported ROS image encoding: {}".format(msg.encoding))
    dtype = dtype_by_encoding[encoding]
    channels = 3 if encoding in {"bgr8", "rgb8"} else 1
    values = np.frombuffer(msg.data, dtype=dtype)
    image = values.reshape(msg.height, msg.width, channels) if channels == 3 else values.reshape(msg.height, msg.width)
    if not is_depth and encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def decode_image(msg, is_depth=False):
    if hasattr(msg, "format"):
        return decode_compressed_image(msg, is_depth)
    if hasattr(msg, "encoding"):
        return decode_raw_image(msg, is_depth)
    raise ValueError("Unsupported image message type: {}".format(getattr(msg, "_type", type(msg).__name__)))


def decode_livox_points(msg):
    points = getattr(msg, "points", None)
    if points is None:
        return None
    xyzi = np.empty((len(points), 4), dtype=np.float32)
    for idx, point in enumerate(points):
        xyzi[idx] = [
            float(point.x),
            float(point.y),
            float(point.z),
            float(getattr(point, "reflectivity", getattr(point, "intensity", 0.0))),
        ]
    return xyzi


def decode_pointcloud2(msg):
    try:
        from sensor_msgs import point_cloud2
    except ImportError as exc:
        raise ImportError("sensor_msgs.point_cloud2 is required to decode PointCloud2 lidar messages") from exc

    field_names = {field.name for field in msg.fields}
    intensity_field = "intensity" if "intensity" in field_names else "reflectivity" if "reflectivity" in field_names else None
    requested = ["x", "y", "z"] + ([intensity_field] if intensity_field else [])
    points = list(point_cloud2.read_points(msg, field_names=requested, skip_nans=True))
    xyzi = np.zeros((len(points), 4), dtype=np.float32)
    if points:
        values = np.asarray(points, dtype=np.float32)
        xyzi[:, :3] = values[:, :3]
        if intensity_field:
            xyzi[:, 3] = values[:, 3]
    return xyzi


def decode_lidar(msg):
    xyzi = decode_livox_points(msg)
    if xyzi is not None:
        return xyzi
    if getattr(msg, "_type", "") == "sensor_msgs/PointCloud2":
        return decode_pointcloud2(msg)
    raise ValueError("Unsupported lidar message type: {}".format(getattr(msg, "_type", type(msg).__name__)))


def prepare_output_dirs(output_dir, overwrite):
    if output_dir.exists() and not overwrite:
        raise FileExistsError("{} already exists; pass --overwrite to replace files".format(output_dir))
    for modality in ("audio", "image", "depth", "lidar"):
        (output_dir / modality).mkdir(parents=True, exist_ok=True)


def write_sample(output_dir, index, sample_rate, audio, color_msg, depth_msg, lidar_msg):
    stem = "{:04d}".format(index)
    wavfile.write(str(output_dir / "audio" / (stem + ".wav")), sample_rate, audio)
    if not cv2.imwrite(str(output_dir / "image" / (stem + ".png")), decode_image(color_msg)):
        raise IOError("Failed to write color image {}".format(stem))
    if not cv2.imwrite(str(output_dir / "depth" / (stem + ".png")), decode_image(depth_msg, is_depth=True)):
        raise IOError("Failed to write depth image {}".format(stem))
    decode_lidar(lidar_msg).astype(np.float32).tofile(str(output_dir / "lidar" / (stem + ".bin")))


def process_job(job, args):
    bag_path, wav_path, first_stamp_ns = job
    output_dir = args.output_root / bag_path.stem

    sample_rate, audio = wavfile.read(str(wav_path))
    if audio.ndim == 1:
        audio = audio[:, None]
    segment_frames = int(round(sample_rate * args.segment_seconds))
    hop_seconds = args.segment_seconds if args.hop_seconds is None else args.hop_seconds
    hop_frames = int(round(sample_rate * hop_seconds))
    if segment_frames <= 0:
        raise ValueError("segment-seconds must produce at least one audio frame")
    if hop_frames <= 0:
        raise ValueError("hop-seconds must produce at least one audio frame")
    segment_count = (
        0 if len(audio) < segment_frames else 1 + (len(audio) - segment_frames) // hop_frames
    )

    topics = (args.color_topic, args.depth_topic, args.lidar_topic)
    messages = load_sensor_messages(bag_path, topics, args.header_stamp)
    prepare_output_dirs(output_dir, args.overwrite)
    max_gap_ns = int(round(args.max_sync_gap_seconds * 1e9))
    saved = 0
    with (output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "source_wav",
                "segment_start_ns",
                "segment_center_ns",
                "color_stamp_ns",
                "depth_stamp_ns",
                "lidar_stamp_ns",
            ]
        )
        for segment_idx in range(segment_count):
            start_frame = segment_idx * hop_frames
            end_frame = start_frame + segment_frames
            start_ns = first_stamp_ns + int(round(segment_idx * hop_seconds * 1e9))
            center_ns = start_ns + int(round(0.5 * args.segment_seconds * 1e9))
            color = nearest_message(messages[args.color_topic], center_ns, max_gap_ns)
            depth = nearest_message(messages[args.depth_topic], center_ns, max_gap_ns)
            lidar = nearest_message(messages[args.lidar_topic], center_ns, max_gap_ns)
            if color is None or depth is None or lidar is None:
                continue
            saved += 1
            write_sample(
                output_dir,
                saved,
                sample_rate,
                audio[start_frame:end_frame],
                color[1],
                depth[1],
                lidar[1],
            )
            writer.writerow(
                [f"{saved:04d}", str(wav_path), start_ns, center_ns, color[0], depth[0], lidar[0]]
            )
    return saved


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = match_wavs_to_bags(args.wav_dir, args.bag_dir, args.timestamp_unit, args.recursive, args.bag_name)
    if not jobs:
        raise RuntimeError("No timestamped wav files matched ROS 1 bag files.")
    for job in jobs:
        bag_path, wav_path, _ = job
        try:
            saved = process_job(job, args)
            print("[OK] {} <- {}: {} samples".format(bag_path.stem, wav_path.name, saved))
        except Exception as exc:
            print("[ERROR] {} <- {}: {}".format(bag_path.stem, wav_path.name, exc))


if __name__ == "__main__":
    main()
