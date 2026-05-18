from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

from preprocess.audio_process import Audio2Spectrogram
from preprocess.image_process import preprocess_input


class AVpedPairsLoader(Dataset):
    """Dataset loader for the split/bag/modalities structure under data/pairs."""

    def __init__(
        self,
        root_path="data/pairs",
        split="train",
        audio_channels=(0, 1, 2, 3),
        image_size=(256, 256),
        depth_size=(256, 256),
        include_lidar=False,
        return_metadata=False,
        skip_empty_labels=True,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.split_root = self.root_path / split
        self.audio_channels = audio_channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.include_lidar = include_lidar
        self.return_metadata = return_metadata
        self.skip_empty_labels = skip_empty_labels
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for bag_dir in sorted(path for path in self.split_root.iterdir() if path.is_dir()):
            for audio_path in sorted((bag_dir / "audio").glob("*.wav")):
                stem = audio_path.stem
                sample = {
                    "split": self.split,
                    "bag": bag_dir.name,
                    "stem": stem,
                    "audio": audio_path,
                    "image": bag_dir / "image" / f"{stem}.png",
                    "depth": bag_dir / "depth" / f"{stem}.png",
                    "lidar": bag_dir / "lidar" / f"{stem}.bin",
                    "label": bag_dir / "labels" / f"{stem}.txt",
                }
                if not all(sample[key].exists() for key in ("image", "depth", "lidar", "label")):
                    continue
                if self.skip_empty_labels and not sample["label"].read_text().strip():
                    continue
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_label(path):
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) == 0:
            return np.zeros(7, dtype=np.float32)
        if len(lines) != 1:
            raise ValueError(f"Expected one bbox in {path}, got {len(lines)}")

        parts = lines[0].split()
        if len(parts) != 15:
            raise ValueError(f"Expected 15 label fields in {path}, got {len(parts)}")
        h, w, l = map(float, parts[8:11])
        x, y, z, yaw = map(float, parts[11:15])
        return np.asarray([x, y, z, l, w, h, yaw], dtype=np.float32)

    def _load_audio_spec(self, path):
        sample_rate, audio = wavfile.read(path)
        if audio.ndim == 1:
            audio = audio[:, None]
        audio = audio.astype(np.float32)
        if self.audio_channels is not None:
            if max(self.audio_channels) >= audio.shape[1]:
                raise ValueError(
                    f"Requested audio channels {self.audio_channels}, "
                    f"but {path} only has {audio.shape[1]} channels"
                )
            audio = audio[:, self.audio_channels]

        audio = np.transpose(audio, [1, 0])
        return Audio2Spectrogram(audio, sr=sample_rate).float()

    def _load_image(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = preprocess_input(image)
        image = np.transpose(image, [2, 0, 1])
        return torch.from_numpy(image).float()

    def _load_depth(self, path):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)
        depth = cv2.resize(depth, self.depth_size, interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32)
        if depth.max() > 0:
            depth = depth / depth.max()
        return torch.from_numpy(depth[None, ...]).float()

    @staticmethod
    def _load_lidar(path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def __getitem__(self, index):
        sample = self.samples[index]
        spec = self._load_audio_spec(sample["audio"])
        image = self._load_image(sample["image"])
        depth = self._load_depth(sample["depth"])
        gt = torch.from_numpy(self._load_label(sample["label"])).float()

        output = [spec, image, depth, gt]
        if self.include_lidar:
            output.append(self._load_lidar(sample["lidar"]))
        if self.return_metadata:
            output.append(
                {
                    "split": sample["split"],
                    "bag": sample["bag"],
                    "stem": sample["stem"],
                }
            )
        return tuple(output)
