from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from modules.core_components.tools.live_audio_streaming import (
    LiveAudioChunkWriter,
    normalize_audio_chunk,
)


def test_normalize_audio_chunk_coerces_to_mono_float32():
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int16)
    out = normalize_audio_chunk(arr)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] == 6


def test_live_audio_chunk_writer_finalize_and_duration(tmp_path):
    final_path = tmp_path / "preview.wav"
    writer = LiveAudioChunkWriter(final_path)

    c1 = writer.write_chunk(np.array([0.1, -0.1, 0.0], dtype=np.float32), 24000)
    c2 = writer.write_chunk(np.array([0.2, 0.3], dtype=np.float32), 24000)
    assert c1.endswith(".wav")
    assert c2.endswith(".wav")

    writer.add_silence(0.1)
    result = writer.finalize()

    assert result.path == str(final_path)
    assert result.sample_rate == 24000
    assert result.chunk_count == 2
    assert result.total_samples > 5
    assert result.duration_seconds == pytest.approx(result.total_samples / 24000.0, rel=1e-5)

    data, sr = sf.read(result.path)
    assert sr == 24000
    assert len(data) == result.total_samples

