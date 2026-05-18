from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset
import torchaudio.transforms as T

from preprocess.audio_process import Audio2IPDSpectrogram, Audio2Spectrogram


class AVpedAudioAngleLoader(Dataset):
    """Audio-only dataset with azimuth targets derived from bbox center x/y."""

    def __init__(
        self,
        root_path="data/pairs",
        split="train",
        audio_channels=(0, 1, 2, 3),
        feature_type="ipd",
        augment=False,
        freq_mask_param=8,
        time_mask_param=8,
        return_angle=False,
        return_metadata=False,
        skip_empty_labels=True,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.split_root = self.root_path / split
        self.audio_channels = audio_channels
        self.feature_type = feature_type
        self.augment = augment
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.return_angle = return_angle
        self.return_metadata = return_metadata
        self.skip_empty_labels = skip_empty_labels
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for bag_dir in sorted(path for path in self.split_root.iterdir() if path.is_dir()):
            for audio_path in sorted((bag_dir / "audio").glob("*.wav")):
                stem = audio_path.stem
                label_path = bag_dir / "labels" / f"{stem}.txt"
                if not label_path.exists():
                    continue
                if self.skip_empty_labels and not label_path.read_text().strip():
                    continue
                samples.append(
                    {
                        "split": self.split,
                        "bag": bag_dir.name,
                        "stem": stem,
                        "audio": audio_path,
                        "label": label_path,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

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
        if self.feature_type == "mel":
            spec = Audio2Spectrogram(audio, sr=sample_rate).float()
        elif self.feature_type == "ipd":
            spec = Audio2IPDSpectrogram(audio, sr=sample_rate).float()
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        if self.augment:
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)
        return spec

    @staticmethod
    def _load_azimuth(path):
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(f"Expected one bbox in {path}, got {len(lines)}")
        parts = lines[0].split()
        if len(parts) != 15:
            raise ValueError(f"Expected 15 label fields in {path}, got {len(parts)}")
        x, y = map(float, parts[11:13])
        angle = np.arctan2(y, x).astype(np.float32)
        vector = np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32)
        return angle, vector

    def __getitem__(self, index):
        sample = self.samples[index]
        spec = self._load_audio_spec(sample["audio"])
        angle, vector = self._load_azimuth(sample["label"])
        output = [spec, torch.from_numpy(vector).float()]
        if self.return_angle:
            output.append(torch.tensor(angle).float())
        if self.return_metadata:
            output.append(
                {
                    "split": sample["split"],
                    "bag": sample["bag"],
                    "stem": sample["stem"],
                }
            )
        return tuple(output)
