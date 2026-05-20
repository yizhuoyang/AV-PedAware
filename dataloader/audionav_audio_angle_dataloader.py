from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T
from scipy.io import wavfile
from torch.utils.data import Dataset

from preprocess.audio_process import Audio2IPDSpectrogram, Audio2Spectrogram


class AudioNavAudioAngleLoader(Dataset):
    """Audio-only DOA loader for audio-nav synced_dataset."""

    def __init__(
        self,
        root_path="/home/kemove/yyz/audio-nav/ws_col/synced_dataset",
        split="train",
        audio_channels=(0, 1, 2, 3),
        feature_type="ipd",
        doa_field="heading_target_yaw_signed_deg",
        object_filter=None,
        augment=False,
        freq_mask_param=8,
        time_mask_param=8,
        return_angle=False,
        return_metadata=False,
        sample_rate=16000,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.audio_channels = tuple(audio_channels)
        self.feature_type = feature_type
        self.doa_field = doa_field
        self.object_filter = object_filter
        self.augment = augment
        self.return_angle = return_angle
        self.return_metadata = return_metadata
        self.sample_rate = sample_rate
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.selected_sequences = []
        self.samples = self._collect_samples()

    def _sequence_root(self):
        split_root = self.root_path / self.split
        if split_root.exists() and split_root.is_dir():
            return split_root
        if self.split == "val" and (self.root_path / "test").exists():
            return self.root_path / "test"
        return self.root_path

    def _list_sequences(self):
        seqs = [p for p in sorted(self._sequence_root().iterdir()) if p.is_dir()]
        if self.object_filter:
            prefixes = tuple(prefix.strip() for prefix in self.object_filter.split(",") if prefix.strip())
            seqs = [p for p in seqs if p.name.startswith(prefixes)]
            if not seqs:
                raise ValueError(f"No sequences found for object_filter={self.object_filter}")
        self.selected_sequences = [p.name for p in seqs]
        return seqs

    def _load_doa_table(self, seq_dir):
        npz_path = seq_dir / "doa_lio_odom.npz"
        if not npz_path.exists():
            return {}
        data = np.load(npz_path, allow_pickle=True)
        fields = [str(field) for field in data["fields"]]
        if self.doa_field not in fields:
            raise KeyError(f"{self.doa_field} not found in {npz_path}; available fields: {fields}")
        field_idx = fields.index(self.doa_field)
        return {
            str(sample_id): np.float32(row[field_idx] % 360.0)
            for sample_id, row in zip(data["sample_ids"], data["data"])
        }

    def _collect_samples(self):
        samples = []
        for seq_dir in self._list_sequences():
            doa_by_id = self._load_doa_table(seq_dir)
            for sample_id, doa in sorted(doa_by_id.items()):
                audio_path = seq_dir / "audio" / f"{sample_id}.wav"
                if not audio_path.exists():
                    continue
                samples.append(
                    {
                        "split": self.split,
                        "sequence": seq_dir.name,
                        "stem": sample_id,
                        "audio": audio_path,
                        "doa": doa,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_audio_spec(self, path):
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

    def __getitem__(self, index):
        sample = self.samples[index]
        spec = self._load_audio_spec(sample["audio"])
        angle = np.deg2rad(sample["doa"]).astype(np.float32)
        vector = np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32)
        output = [spec, torch.from_numpy(vector).float()]
        if self.return_angle:
            output.append(torch.tensor(angle).float())
        if self.return_metadata:
            output.append(
                {
                    "split": sample["split"],
                    "sequence": sample["sequence"],
                    "stem": sample["stem"],
                }
            )
        return tuple(output)
