from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

import modules.core_components.ai_models.tts_manager as tts_manager_mod
from modules.core_components.ai_models.tts_manager import TTSManager


def test_supports_live_streaming_matrix(monkeypatch):
    mgr = TTSManager(user_config={})

    class _PatchState:
        available = True

    monkeypatch.setattr(mgr, "_ensure_qwen_stream_patch", lambda: _PatchState())

    assert mgr.supports_live_streaming("qwen", "voice_clone") is True
    assert mgr.supports_live_streaming("qwen", "conversation") is True
    assert mgr.supports_live_streaming("qwen", "voice_design") is False
    assert mgr.supports_live_streaming("vibevoice", "voice_clone") is True
    assert mgr.supports_live_streaming("vibevoice", "conversation") is True
    assert mgr.supports_live_streaming("luxtts", "voice_clone") is False


def test_normalize_stream_chunk_handles_bfloat16_tensor():
    chunk = torch.tensor([[0.1, -0.2]], dtype=torch.bfloat16)

    arr = TTSManager._normalize_stream_chunk(chunk)

    assert arr.dtype == np.float32
    assert arr.shape == (2,)
    np.testing.assert_allclose(arr, np.array([0.1, -0.2], dtype=np.float32), rtol=1e-2, atol=1e-2)


def test_qwen_runtime_streaming_available_false_when_model_missing(monkeypatch):
    mgr = TTSManager(user_config={})

    class _PatchState:
        available = True

    monkeypatch.setattr(mgr, "_ensure_qwen_stream_patch", lambda: _PatchState())
    monkeypatch.setattr(mgr, "get_qwen3_base", lambda _size: (_ for _ in ()).throw(RuntimeError("load fail")))

    assert mgr.qwen_runtime_streaming_available("1.7B") is False


def test_stream_voice_clone_qwen_unavailable_raises(monkeypatch):
    mgr = TTSManager(user_config={})

    class _PatchState:
        available = False
        reason = "missing"

    monkeypatch.setattr(mgr, "_ensure_qwen_stream_patch", lambda: _PatchState())
    with pytest.raises(RuntimeError, match="Qwen live streaming unavailable"):
        next(
            mgr.stream_voice_clone_qwen(
                text="hello",
                language="Auto",
                prompt_items={"x": 1},
            )
        )


def test_stream_voice_clone_vibevoice_adapter_yields_chunks(monkeypatch):
    mgr = TTSManager(user_config={})

    class FakeProcessor:
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

        def __call__(self, **kwargs):
            del kwargs
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class FakeStreamer:
        def __init__(self, batch_size, stop_signal=None, timeout=None):
            del batch_size, timeout
            self.stop_signal = stop_signal if stop_signal is not None else object()
            import queue

            self.q = queue.Queue()

        def put(self, audio_chunks, sample_indices):
            for i, sample_idx in enumerate(sample_indices):
                _ = sample_idx  # keep shape-compatible API
                self.q.put(audio_chunks[i])

        def end(self, _sample_indices=None):
            self.q.put(self.stop_signal)

        def get_stream(self, _sample_idx):
            while True:
                value = self.q.get()
                if value is self.stop_signal:
                    break
                yield value

    class FakeModel:
        def set_ddpm_inference_steps(self, _num_steps):
            return None

        def generate(self, **kwargs):
            streamer = kwargs["audio_streamer"]
            streamer.put(np.array([[0.1, 0.2]], dtype=np.float32), np.array([0]))
            streamer.put(np.array([[0.3]], dtype=np.float32), np.array([0]))
            streamer.end()

    fake_processor_mod = types.ModuleType("modules.vibevoice_tts.processor.vibevoice_processor")
    fake_processor_mod.VibeVoiceProcessor = FakeProcessor
    fake_streamer_mod = types.ModuleType("modules.vibevoice_tts.modular.streamer")
    fake_streamer_mod.AudioStreamer = FakeStreamer

    monkeypatch.setitem(sys.modules, "modules.vibevoice_tts.processor.vibevoice_processor", fake_processor_mod)
    monkeypatch.setitem(sys.modules, "modules.vibevoice_tts.modular.streamer", fake_streamer_mod)

    monkeypatch.setattr(tts_manager_mod, "resolve_model_source", lambda *args, **kwargs: "local")
    monkeypatch.setattr(tts_manager_mod, "get_device", lambda: "cpu")
    monkeypatch.setattr(mgr, "get_vibevoice_tts", lambda _size: FakeModel())

    chunks = list(
        mgr.stream_voice_clone_vibevoice(
            text="hello world",
            voice_sample_path="/tmp/sample.wav",
            model_size="Large",
            user_config={},
        )
    )

    assert len(chunks) >= 1
    assert all(sr == 24000 for _, sr in chunks)
    merged = np.concatenate([arr for arr, _ in chunks], axis=0)
    np.testing.assert_allclose(merged, np.array([0.1, 0.2, 0.3], dtype=np.float32), rtol=1e-6, atol=1e-6)
