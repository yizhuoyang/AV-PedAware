#!/usr/bin/env python3
"""Build timestamp-aligned audio/image/depth/lidar samples from ROS 2 bags."""

import argparse
import csv
import sqlite3
import struct
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from scipy.io import wavfile
from sensor_msgs.msg import CompressedImage


COLOR_TOPIC = "/camera/color/image_raw/compressed"
DEPTH_TOPIC = "/camera/depth/image_raw/compressedDepth"
LIDAR_TOPIC = "/livox/lidar"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestamps-csv",
        type=Path,
        default=Path("data/wav/audio_timestamps.csv"),
        help="CSV produced when extracting wav files from bags.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/pairs"),
        help="Root directory for paired samples.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=0.5,
        help="Audio segment duration in seconds.",
    )
    parser.add_argument(
        "--max-sync-gap-seconds",
        type=float,
        default=0.25,
        help="Maximum allowed timestamp gap when selecting the nearest sensor frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs for a bag.",
    )
    parser.add_argument(
        "--bag-name",
        help="Only process one bag directory name, for example rosbag2_2026_05_17-09_34_07.",
    )
    return parser.parse_args()


def load_rows(csv_path):
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def get_topic_ids(conn):
    rows = conn.execute("SELECT id, name FROM topics").fetchall()
    topic_ids = {name: topic_id for topic_id, name in rows}
    required = [COLOR_TOPIC, DEPTH_TOPIC, LIDAR_TOPIC]
    missing = [topic for topic in required if topic not in topic_ids]
    if missing:
        raise ValueError("Missing required topics: {}".format(", ".join(missing)))
    return topic_ids


def read_topic_rows(conn, topic_id):
    return conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    ).fetchall()


def nearest_message(rows, timestamps, target_ns, max_gap_ns):
    idx = bisect_left(timestamps, target_ns)
    candidates = []
    if idx < len(rows):
        candidates.append(rows[idx])
    if idx > 0:
        candidates.append(rows[idx - 1])
    if not candidates:
        return None
    selected = min(candidates, key=lambda row: abs(row[0] - target_ns))
    if abs(selected[0] - target_ns) > max_gap_ns:
        return None
    return selected


def decode_color(blob):
    msg = deserialize_message(blob, CompressedImage)
    encoded = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode color image")
    return image


def decode_depth(blob):
    msg = deserialize_message(blob, CompressedImage)
    data = bytes(msg.data)
    if "compressedDepth" not in msg.format:
        raise ValueError("Unexpected depth format: {}".format(msg.format))
    # compressedDepth prepends a 12-byte codec header before the PNG payload.
    encoded = np.frombuffer(data[12:], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode depth image")
    return image


def decode_livox_custom_msg(blob):
    """Decode livox_ros_driver2/msg/CustomMsg into xyzi float32 points.

    Message layout matches the public Livox CustomMsg definition:
    Header, uint64 timebase, uint32 point_num, uint8 lidar_id, uint8[3] rsvd,
    CustomPoint[] points. Each point is packed as uint32 offset_time,
    float32 x/y/z, uint8 reflectivity/tag/line plus one byte of alignment.
    """
    offset = 4  # CDR encapsulation header
    offset += 8  # std_msgs/Header stamp

    frame_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4 + frame_len

    _, point_num = struct.unpack_from("<QI", blob, offset)
    offset += 12
    offset += 4  # lidar_id + rsvd[3]
    seq_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    count = min(point_num, seq_len)
    point_dtype = np.dtype(
        [
            ("offset_time", "<u4"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("reflectivity", "u1"),
            ("tag", "u1"),
            ("line", "u1"),
            ("padding", "u1"),
        ]
    )
    available = max(0, (len(blob) - offset) // point_dtype.itemsize)
    count = min(count, available)
    points = np.frombuffer(blob, dtype=point_dtype, count=count, offset=offset)

    xyzi = np.empty((count, 4), dtype=np.float32)
    xyzi[:, 0] = points["x"]
    xyzi[:, 1] = points["y"]
    xyzi[:, 2] = points["z"]
    xyzi[:, 3] = points["reflectivity"].astype(np.float32)
    return xyzi


def prepare_bag_dirs(bag_dir, overwrite):
    if bag_dir.exists() and not overwrite:
        raise FileExistsError(
            "{} already exists; pass --overwrite to replace files".format(bag_dir)
        )
    for name in ("audio", "image", "depth", "lidar"):
        (bag_dir / name).mkdir(parents=True, exist_ok=True)


def save_segment_outputs(
    bag_dir,
    index,
    sample_rate,
    audio_segment,
    color_blob,
    depth_blob,
    lidar_blob,
):
    stem = "{:04d}".format(index)
    wavfile.write(str(bag_dir / "audio" / "{}.wav".format(stem)), sample_rate, audio_segment)
    cv2.imwrite(str(bag_dir / "image" / "{}.png".format(stem)), decode_color(color_blob))
    cv2.imwrite(str(bag_dir / "depth" / "{}.png".format(stem)), decode_depth(depth_blob))
    decode_livox_custom_msg(lidar_blob).tofile(str(bag_dir / "lidar" / "{}.bin".format(stem)))


def process_row(row, output_root, segment_seconds, max_gap_ns, overwrite):
    wav_path = Path(row["output_wav"])
    if not wav_path.exists():
        return None, "missing wav: {}".format(wav_path)

    bag_path = Path(row["bag"])
    db_path = Path(row["sqlite_database"])
    bag_name = bag_path.name
    bag_dir = output_root / bag_name
    prepare_bag_dirs(bag_dir, overwrite)

    sample_rate, audio = wavfile.read(str(wav_path))
    if audio.ndim == 1:
        audio = audio[:, None]
    segment_frames = int(round(sample_rate * segment_seconds))
    segment_count = len(audio) // segment_frames

    conn = sqlite3.connect(str(db_path))
    topic_ids = get_topic_ids(conn)
    color_rows = read_topic_rows(conn, topic_ids[COLOR_TOPIC])
    depth_rows = read_topic_rows(conn, topic_ids[DEPTH_TOPIC])
    lidar_rows = read_topic_rows(conn, topic_ids[LIDAR_TOPIC])
    conn.close()

    color_ts = [row[0] for row in color_rows]
    depth_ts = [row[0] for row in depth_rows]
    lidar_ts = [row[0] for row in lidar_rows]
    first_stamp_ns = int(row["first_stamp_ns"])
    manifest_path = bag_dir / "manifest.csv"

    saved = 0
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample",
                "segment_start_ns",
                "segment_center_ns",
                "color_stamp_ns",
                "depth_stamp_ns",
                "lidar_stamp_ns",
            ]
        )
        for index in range(segment_count):
            start_frame = index * segment_frames
            end_frame = start_frame + segment_frames
            center_ns = first_stamp_ns + int((index + 0.5) * segment_seconds * 1e9)
            start_ns = first_stamp_ns + int(index * segment_seconds * 1e9)

            color = nearest_message(color_rows, color_ts, center_ns, max_gap_ns)
            depth = nearest_message(depth_rows, depth_ts, center_ns, max_gap_ns)
            lidar = nearest_message(lidar_rows, lidar_ts, center_ns, max_gap_ns)
            if color is None or depth is None or lidar is None:
                continue

            saved += 1
            save_segment_outputs(
                bag_dir,
                saved,
                sample_rate,
                audio[start_frame:end_frame],
                color[1],
                depth[1],
                lidar[1],
            )
            writer.writerow(
                [
                    "{:04d}".format(saved),
                    start_ns,
                    center_ns,
                    color[0],
                    depth[0],
                    lidar[0],
                ]
            )
    return saved, None


def main():
    args = parse_args()
    max_gap_ns = int(args.max_sync_gap_seconds * 1e9)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for row in load_rows(args.timestamps_csv):
        bag_name = Path(row["bag"]).name
        if args.bag_name and bag_name != args.bag_name:
            continue
        try:
            saved, error = process_row(
                row,
                args.output_root,
                args.segment_seconds,
                max_gap_ns,
                args.overwrite,
            )
        except Exception as exc:
            print("[ERROR] {}: {}".format(bag_name, exc))
            continue

        if error:
            print("[SKIP] {}: {}".format(bag_name, error))
        else:
            print("[OK] {}: {} samples".format(bag_name, saved))


if __name__ == "__main__":
    main()
