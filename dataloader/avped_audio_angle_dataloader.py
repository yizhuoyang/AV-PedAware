from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset
import torchaudio.transforms as T

from preprocess.audio_process import Audio2IPDSpectrogram, Audio2Spectrogram


class AVpedAudioAngleLoader(Dataset):
    """Audio-only dataset with azimuth targets derived from bbox center x/y.

    Supports both `data/pairs/<split>/<sequence>` and the current
    `data/pairs_ros1/<split>/<sequence>` layouts.
    """

    def __init__(
        self,
        root_path="data/pairs",
        split="train",
        audio_channels=(0, 1, 2, 3),
        feature_type="ipd",
        augment=False,
        freq_mask_param=8,
        time_mask_param=8,
        object_filter=None,
        sequence_names=None,
        noise_wav=None,
        noise_probability=0.0,
        snr_db_range=(0.0, 20.0),
        low_cut_hz=None,
        high_cut_hz=None,
        filter_order=4,
        num_angle_bins=360,
        target_sigma_deg=5.0,
        return_angle=False,
        return_metadata=False,
        skip_empty_labels=True,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.split_root = self._split_root()
        self.audio_channels = tuple(audio_channels) if audio_channels is not None else None
        self.feature_type = feature_type
        self.augment = augment
        self.object_filter = object_filter
        self.sequence_names = set(sequence_names or [])
        self.noise_wav = Path(noise_wav) if noise_wav else None
        self.noise_probability = float(noise_probability)
        self.snr_db_range = tuple(float(value) for value in snr_db_range)
        if not 0.0 <= self.noise_probability <= 1.0:
            raise ValueError("noise_probability must be between 0 and 1")
        if len(self.snr_db_range) != 2 or self.snr_db_range[0] > self.snr_db_range[1]:
            raise ValueError("snr_db_range must be (min_snr_db, max_snr_db)")
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.filter_order = int(filter_order)
        if self.filter_order <= 0:
            raise ValueError("filter_order must be greater than zero")
        self.num_angle_bins = int(num_angle_bins)
        self.target_sigma_deg = float(target_sigma_deg)
        if self.num_angle_bins <= 1:
            raise ValueError("num_angle_bins must be greater than one")
        if self.target_sigma_deg <= 0.0:
            raise ValueError("target_sigma_deg must be greater than zero")
        self.noise_sample_rate = None
        self.noise_audio = None
        if self.noise_wav is not None:
            self.noise_sample_rate, self.noise_audio = wavfile.read(self.noise_wav)
            if self.noise_audio.ndim == 1:
                self.noise_audio = self.noise_audio[:, None]
            self.noise_audio = self.noise_audio.astype(np.float32)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.return_angle = return_angle
        self.return_metadata = return_metadata
        self.skip_empty_labels = skip_empty_labels
        self.selected_sequences = []
        self.samples = self._collect_samples()

    def _split_root(self):
        split_root = self.root_path / self.split
        if split_root.is_dir():
            return split_root
        if self.split == "val" and (self.root_path / "test").is_dir():
            return self.root_path / "test"
        return self.root_path

    def _sequence_dirs(self):
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"Dataset split directory does not exist: {self.split_root}")
        sequences = [path for path in sorted(self.split_root.iterdir()) if (path / "audio").is_dir()]
        if self.sequence_names:
            sequences = [path for path in sequences if path.name in self.sequence_names]
        if self.object_filter:
            prefixes = tuple(
                name.strip() for name in self.object_filter.split(",") if name.strip()
            )
            sequences = [path for path in sequences if path.name.startswith(prefixes)]
        if not sequences:
            raise ValueError(
                f"No sequences found in {self.split_root} for "
                f"object_filter={self.object_filter or 'all'}"
            )
        self.selected_sequences = [path.name for path in sequences]
        return sequences

    def _collect_samples(self):
        samples = []
        for bag_dir in self._sequence_dirs():
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
                        "image": bag_dir / "image" / f"{stem}.png",
                        "depth": bag_dir / "depth" / f"{stem}.png",
                        "lidar": bag_dir / "lidar" / f"{stem}.bin",
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def get_angle_rad(self, index):
        """Return the bbox-derived azimuth without loading audio features."""
        angle = self._load_azimuth(self.samples[index]["label"])
        return float(angle)

    def _select_audio_channels(self, audio, path):
        if self.audio_channels is None:
            return audio
        if max(self.audio_channels) >= audio.shape[1]:
            raise ValueError(
                f"Requested audio channels {self.audio_channels}, "
                f"but {path} only has {audio.shape[1]} channels"
            )
        return audio[:, self.audio_channels]

    def _mix_random_noise(self, audio, sample_rate):
        if (
            not self.augment
            or self.noise_audio is None
            or np.random.random() >= self.noise_probability
        ):
            return audio
        if sample_rate != self.noise_sample_rate:
            raise ValueError(
                f"Noise sample rate is {self.noise_sample_rate} Hz, "
                f"but input audio is {sample_rate} Hz"
            )

        noise = self._select_audio_channels(self.noise_audio, self.noise_wav)
        if len(noise) < len(audio):
            repeats = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, (repeats, 1))
        max_start = len(noise) - len(audio)
        start = np.random.randint(max_start + 1) if max_start > 0 else 0
        noise = noise[start : start + len(audio)]

        signal_power = float(np.mean(audio ** 2))
        noise_power = float(np.mean(noise ** 2))
        if signal_power <= 0.0 or noise_power <= 0.0:
            return audio
        snr_db = np.random.uniform(*self.snr_db_range)
        scale = np.sqrt(signal_power / (noise_power * (10.0 ** (snr_db / 10.0))))
        return audio + noise * scale

    def _filter_audio(self, audio, sample_rate):
        if self.low_cut_hz is None and self.high_cut_hz is None:
            return audio
        nyquist = sample_rate / 2.0
        low = None if self.low_cut_hz is None else float(self.low_cut_hz)
        high = None if self.high_cut_hz is None else float(self.high_cut_hz)
        if low is not None and not 0.0 < low < nyquist:
            raise ValueError(f"low_cut_hz must be between 0 and {nyquist}, got {low}")
        if high is not None and not 0.0 < high < nyquist:
            raise ValueError(f"high_cut_hz must be between 0 and {nyquist}, got {high}")
        if low is not None and high is not None and low >= high:
            raise ValueError(f"low_cut_hz must be lower than high_cut_hz, got {low} >= {high}")

        if low is not None and high is not None:
            cutoff, filter_type = [low, high], "bandpass"
        elif low is not None:
            cutoff, filter_type = low, "highpass"
        else:
            cutoff, filter_type = high, "lowpass"
        sos = butter(self.filter_order, cutoff, btype=filter_type, fs=sample_rate, output="sos")
        return sosfiltfilt(sos, audio, axis=0).astype(np.float32, copy=False)

    def _load_audio_spec(self, path):
        sample_rate, audio = wavfile.read(path)
        if audio.ndim == 1:
            audio = audio[:, None]
        audio = audio.astype(np.float32)
        audio = self._select_audio_channels(audio, path)
        audio = self._mix_random_noise(audio, sample_rate)
        audio = self._filter_audio(audio, sample_rate)
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
        return np.arctan2(y, x).astype(np.float32)

    def _angle_distribution(self, angle):
        bin_angles = np.arange(self.num_angle_bins, dtype=np.float32) * (
            2.0 * np.pi / self.num_angle_bins
        )
        difference = np.arctan2(np.sin(bin_angles - angle), np.cos(bin_angles - angle))
        sigma = np.deg2rad(self.target_sigma_deg)
        distribution = np.exp(-0.5 * (difference / sigma) ** 2)
        return (distribution / distribution.sum()).astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        spec = self._load_audio_spec(sample["audio"])
        angle = self._load_azimuth(sample["label"])
        distribution = self._angle_distribution(angle)
        output = [spec, torch.from_numpy(distribution).float()]
        if self.return_angle:
            output.append(torch.tensor(angle).float())
        if self.return_metadata:
            output.append(
                {
                    "split": sample["split"],
                    "sequence": sample["bag"],
                    "bag": sample["bag"],
                    "stem": sample["stem"],
                    "audio": str(sample["audio"]),
                    "label": str(sample["label"]),
                    "image": str(sample["image"]),
                    "depth": str(sample["depth"]),
                    "lidar": str(sample["lidar"]),
                }
            )
        return tuple(output)
