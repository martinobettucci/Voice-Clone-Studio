"""Utilities for true live audio streaming previews in generation tabs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def normalize_audio_chunk(audio_chunk: Any) -> np.ndarray:
    """Normalize chunk payloads to mono float32 numpy arrays."""
    if audio_chunk is None:
        return np.zeros(0, dtype=np.float32)

    if hasattr(audio_chunk, "detach") and callable(audio_chunk.detach):
        arr = audio_chunk.detach().cpu().numpy()
    elif hasattr(audio_chunk, "cpu") and hasattr(audio_chunk, "numpy"):
        arr = audio_chunk.cpu().numpy()
    elif hasattr(audio_chunk, "numpy") and callable(audio_chunk.numpy):
        arr = audio_chunk.numpy()
    else:
        arr = np.asarray(audio_chunk)

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


@dataclass(frozen=True)
class LiveAudioFinalizeResult:
    """Summary returned when a live stream preview is finalized."""

    path: str
    sample_rate: int
    duration_seconds: float
    total_samples: int
    chunk_count: int


class LiveAudioChunkWriter:
    """Write streamed chunks to individual WAV previews and one final WAV."""

    def __init__(self, final_output_path: Path, chunk_dir: Path | None = None, chunk_prefix: str | None = None):
        self.final_output_path = Path(final_output_path)
        self.chunk_dir = Path(chunk_dir) if chunk_dir else (self.final_output_path.parent / f"{self.final_output_path.stem}_chunks")
        self.chunk_prefix = chunk_prefix or self.final_output_path.stem

        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

        self._final_file: sf.SoundFile | None = None
        self.sample_rate: int | None = None
        self.total_samples = 0
        self.chunk_count = 0
        self.chunk_paths: list[str] = []

    def _ensure_final_file(self, sample_rate: int) -> None:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        if self._final_file is None:
            self.sample_rate = sample_rate
            self._final_file = sf.SoundFile(
                str(self.final_output_path),
                mode="w",
                samplerate=sample_rate,
                channels=1,
                subtype="PCM_16",
            )
            return

        if self.sample_rate != sample_rate:
            raise ValueError(f"Mismatched sample rates ({self.sample_rate} vs {sample_rate}).")

    def write_chunk(self, audio_chunk: Any, sample_rate: int) -> str:
        """Write one streamed chunk and return its preview filepath."""
        arr = normalize_audio_chunk(audio_chunk)
        if arr.size <= 0:
            return ""

        self._ensure_final_file(sample_rate)
        assert self._final_file is not None

        self._final_file.write(arr)
        self.total_samples += int(arr.shape[0])

        self.chunk_count += 1
        chunk_path = self.chunk_dir / f"{self.chunk_prefix}_chunk_{self.chunk_count:05d}.wav"
        sf.write(str(chunk_path), arr, int(self.sample_rate))
        chunk_path_str = str(chunk_path)
        self.chunk_paths.append(chunk_path_str)
        return chunk_path_str

    def add_silence(self, seconds: float) -> None:
        """Append silence to final output (used for pause stitching)."""
        if self._final_file is None or self.sample_rate is None:
            return

        samples = int(max(0.0, float(seconds)) * int(self.sample_rate))
        if samples <= 0:
            return

        chunk_size = int(self.sample_rate)
        while samples > 0:
            n = min(samples, chunk_size)
            self._final_file.write(np.zeros(n, dtype=np.float32))
            self.total_samples += int(n)
            samples -= n

    def finalize(self) -> LiveAudioFinalizeResult:
        """Close final output and return stream summary."""
        if self._final_file is not None:
            self._final_file.close()
            self._final_file = None

        if self.sample_rate is None or self.total_samples <= 0:
            raise RuntimeError("No streamed audio was written.")

        duration = float(self.total_samples) / float(self.sample_rate)
        return LiveAudioFinalizeResult(
            path=str(self.final_output_path),
            sample_rate=int(self.sample_rate),
            duration_seconds=duration,
            total_samples=int(self.total_samples),
            chunk_count=int(self.chunk_count),
        )

    def cleanup_final_file(self) -> None:
        """Best-effort cleanup of partially written final file."""
        if self._final_file is not None:
            self._final_file.close()
            self._final_file = None
        try:
            if self.final_output_path.exists():
                self.final_output_path.unlink()
        except Exception:
            pass

