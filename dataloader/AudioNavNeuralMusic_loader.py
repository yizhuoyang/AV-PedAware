from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

from dataloader.NeuralMusic_loader import (
    DirectionGrid,
    array_aug,
    compute_correlation_matrices_torch,
)
from dataloader.utils import ModeVector_torch, normalize_magnitude, normalize_phase
import torchvision.transforms as Trans


class AudioNavNeuralMusicLoader(Dataset):
    """NeuralMUSIC loader for synced_dataset from audio-nav.

    Supported layouts:
      synced_dataset/<sequence>/audio/<sample_id>.wav
      synced_dataset/<sequence>/doa_lio_odom.npz

      data/<split>/<sequence>/audio/<sample_id>.wav
      data/<split>/<sequence>/doa/<sample_id>.npy

    Per-sample DOA npy labels use:
      [azimuth_rad, elevation_rad, unit_x, unit_y, unit_z, distance_m]

    The auto target is `heading_target_yaw_signed_deg` for npz labels and
    `azimuth_rad` for per-sample npy labels, both converted to [0, 360).
    """

    ARRAY_DOA_FIELDS = [
        "azimuth_rad",
        "elevation_rad",
        "unit_x",
        "unit_y",
        "unit_z",
        "distance_m",
    ]

    def __init__(
        self,
        root_path="/home/kemove/yyz/audio-nav/ws_col/synced_dataset",
        split="train",
        mic_offsets=None,
        audio_channels=(0, 1, 2, 3),
        feature_type="ipd",
        doa_field="auto",
        train_ratio=0.8,
        sequence_names=None,
        test_sequences=None,
        object_filter=None,
        geometry_aug=False,
        rotation_interval=None,
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.audio_channels = tuple(audio_channels)
        self.feature_type = feature_type
        self.doa_field = doa_field
        self.train_ratio = train_ratio
        self.sequence_names = sequence_names
        self.test_sequences = set(test_sequences or [])
        self.object_filter = object_filter
        self.geometry_aug = geometry_aug
        self.rotation_interval = rotation_interval
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        if mic_offsets is None:
            side = 45.7 / 1000 / 2
            mic_offsets = np.asarray(
                [
                    [side, side, 0.0],
                    [-side, side, 0.0],
                    [-side, -side, 0.0],
                    [side, -side, 0.0],
                ],
                dtype=np.float32,
            )
        self.mic_offsets = np.asarray(mic_offsets, dtype=np.float32)
        if len(self.audio_channels) != len(self.mic_offsets):
            raise ValueError(
                f"audio_channels has {len(self.audio_channels)} entries, "
                f"but mic_offsets has {len(self.mic_offsets)} microphones"
            )

        self.grid = DirectionGrid()
        self.steering_vector = ModeVector_torch(
            self.mic_offsets.T,
            self.sample_rate,
            self.n_fft,
            343,
            self.grid,
            mode="far",
            precompute=True,
        ).mode_vec.to(torch.complex64)
        self.window = torch.hann_window(self.n_fft)
        self.resize = Trans.Resize((self.n_fft // 2 + 1, 64), antialias=True)
        self.selected_sequences = []
        self.samples = self._collect_samples()

    def _split_sequences(self):
        split_root = self.root_path / self.split
        if split_root.exists() and split_root.is_dir():
            seq_root = split_root
        elif self.split == "val" and (self.root_path / "test").exists():
            seq_root = self.root_path / "test"
        else:
            seq_root = self.root_path

        if self.sequence_names:
            requested = set(self.sequence_names)
            seqs = [p for p in sorted(seq_root.iterdir()) if p.is_dir() and p.name in requested]
        else:
            seqs = [p for p in sorted(seq_root.iterdir()) if p.is_dir()]
            if self.object_filter:
                prefixes = tuple(prefix.strip() for prefix in self.object_filter.split(",") if prefix.strip())
                seqs = [p for p in seqs if p.name.startswith(prefixes)]
                if not seqs:
                    raise ValueError(f"No sequences found for object_filter={self.object_filter}")
            if split_root.exists() and split_root.is_dir():
                pass
            elif self.test_sequences:
                missing = self.test_sequences - {p.name for p in seqs}
                if missing:
                    raise ValueError(f"test_sequences not found after filtering: {sorted(missing)}")
                if self.split == "train":
                    seqs = [p for p in seqs if p.name not in self.test_sequences]
                elif self.split in {"val", "test"}:
                    seqs = [p for p in seqs if p.name in self.test_sequences]
                else:
                    raise ValueError(f"Unsupported split: {self.split}")
            else:
                cut = max(1, int(len(seqs) * self.train_ratio))
                if self.split == "train":
                    seqs = seqs[:cut]
                elif self.split in {"val", "test"}:
                    seqs = seqs[cut:]
                else:
                    raise ValueError(f"Unsupported split: {self.split}")
        self.selected_sequences = [p.name for p in seqs]
        return seqs

    def _load_doa_table(self, seq_dir):
        npz_path = seq_dir / "doa_lio_odom.npz"
        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            fields = [str(field) for field in data["fields"]]
            doa_field = (
                "heading_target_yaw_signed_deg"
                if self.doa_field == "auto"
                else self.doa_field
            )
            if doa_field not in fields:
                raise KeyError(f"{doa_field} not found in {npz_path}; available fields: {fields}")
            field_idx = fields.index(doa_field)
            return {
                str(sample_id): np.float32(row[field_idx] % 360.0)
                for sample_id, row in zip(data["sample_ids"], data["data"])
            }

        doa_dir = seq_dir / "doa"
        if doa_dir.exists():
            doa_field = "azimuth_rad" if self.doa_field == "auto" else self.doa_field
            if doa_field not in self.ARRAY_DOA_FIELDS:
                raise KeyError(
                    f"{doa_field} is not a supported per-sample DOA field; "
                    f"available fields: {self.ARRAY_DOA_FIELDS}"
                )
            field_idx = self.ARRAY_DOA_FIELDS.index(doa_field)
            doa_by_id = {}
            for doa_path in sorted(doa_dir.glob("*.npy")):
                values = np.load(doa_path).astype(np.float32).reshape(-1)
                if values.shape[0] <= field_idx:
                    continue
                value = float(values[field_idx])
                if doa_field.endswith("_rad"):
                    value = np.degrees(value)
                doa_by_id[doa_path.stem] = np.float32(value % 360.0)
            return doa_by_id

        return {}

    def _collect_samples(self):
        samples = []
        for seq_dir in self._split_sequences():
            doa_by_id = self._load_doa_table(seq_dir)
            for sample_id, doa in sorted(doa_by_id.items()):
                audio_path = seq_dir / "audio" / f"{sample_id}.wav"
                if not audio_path.exists():
                    continue
                samples.append(
                    {
                        "sequence": seq_dir.name,
                        "stem": sample_id,
                        "audio": audio_path,
                        "doa": doa,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path):
        sample_rate, audio = wavfile.read(path)
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate} Hz in {path}")
        if audio.ndim == 1:
            audio = audio[:, None]
        if max(self.audio_channels) >= audio.shape[1]:
            raise ValueError(
                f"Requested audio channels {self.audio_channels}, but {path} only has {audio.shape[1]} channels"
            )
        audio = audio[:, self.audio_channels].astype(np.float32)
        scale = np.max(np.abs(audio))
        if scale > 0:
            audio = audio / scale
        return torch.from_numpy(audio.T)

    def _spectrogram_process(self, audio):
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        correlation = compute_correlation_matrices_torch(stft)
        if self.feature_type == "magphase":
            magnitude = normalize_magnitude(torch.abs(stft))
            phase = normalize_phase(torch.angle(stft))
            spectrogram = torch.stack([magnitude, phase], dim=1).view(
                2 * audio.shape[0], stft.shape[1], stft.shape[2]
            )
        elif self.feature_type == "ipd":
            magnitude = normalize_magnitude(torch.abs(stft[0:1]))
            ref_phase = torch.angle(stft[0])
            features = [magnitude[0]]
            for channel in range(1, stft.shape[0]):
                phase_diff = torch.angle(stft[channel]) - ref_phase
                features.append(torch.sin(phase_diff))
                features.append(torch.cos(phase_diff))
            spectrogram = torch.stack(features, dim=0)
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        return self.resize(spectrogram).float(), correlation

    def __getitem__(self, index):
        sample = self.samples[index]
        audio = self._load_audio(sample["audio"])
        spectrogram, correlation = self._spectrogram_process(audio)
        doa = np.asarray([sample["doa"]], dtype=np.float32)
        steering_vector, doa = array_aug(
            self.mic_offsets,
            doa,
            transform=self.geometry_aug and self.split == "train",
            grid=self.grid,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            interval=self.rotation_interval,
        )
        return spectrogram, torch.from_numpy(doa).float(), steering_vector.to(torch.complex64), correlation
