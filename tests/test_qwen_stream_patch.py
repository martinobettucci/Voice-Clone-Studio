from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from modules.core_components.ai_models.qwen_stream_patch import (
    apply_qwen_streaming_patch,
    qwen_streaming_runtime_available,
)


def _install_fake_qwen_module(monkeypatch, qwen_class):
    qwen_pkg = types.ModuleType("qwen_tts")
    inference_pkg = types.ModuleType("qwen_tts.inference")
    model_mod = types.ModuleType("qwen_tts.inference.qwen3_tts_model")
    model_mod.Qwen3TTSModel = qwen_class

    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_pkg)
    monkeypatch.setitem(sys.modules, "qwen_tts.inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "qwen_tts.inference.qwen3_tts_model", model_mod)


def test_patch_adds_stream_methods(monkeypatch):
    class FakeInner:
        def stream_generate(self, *args, **kwargs):
            del args, kwargs
            return iter([(np.array([0.1, -0.1], dtype=np.float32), 24000)])

    class FakeQwen:
        def __init__(self):
            self.model = FakeInner()

        def _convert_inputs_2_neural_method_inputs(self, _kwargs, model_type="voice_clone"):
            assert model_type == "voice_clone"
            return {
                "input_ids": np.array([[1, 2, 3]]),
                "attention_mask": np.array([[1, 1, 1]]),
                "speaker_embedding": None,
                "text_tokens_in_input_ids": None,
            }

        def create_voice_clone_prompt(self, **kwargs):
            del kwargs
            return {"ok": True}

    _install_fake_qwen_module(monkeypatch, FakeQwen)
    state = apply_qwen_streaming_patch()

    assert state.available is True
    assert hasattr(FakeQwen, "stream_generate_voice_clone")
    assert hasattr(FakeQwen, "stream_generate_pcm")
    assert hasattr(FakeQwen, "enable_streaming_optimizations")

    model = FakeQwen()
    model.enable_streaming_optimizations(foo="bar")
    chunks = list(
        model.stream_generate_voice_clone(
            text="hello",
            language="Auto",
            voice_clone_prompt={"p": 1},
        )
    )
    assert len(chunks) == 1
    assert chunks[0][1] == 24000
    assert qwen_streaming_runtime_available(model) is True


def test_patch_is_noop_when_methods_exist(monkeypatch):
    class FakeInner:
        def stream_generate(self, *args, **kwargs):
            del args, kwargs
            return iter([])

    class FakeQwen:
        def __init__(self):
            self.model = FakeInner()

        def enable_streaming_optimizations(self, **kwargs):
            return kwargs

        def stream_generate_pcm(self, **kwargs):
            del kwargs
            return iter([])

        def stream_generate_voice_clone(self, **kwargs):
            del kwargs
            return iter([])

    _install_fake_qwen_module(monkeypatch, FakeQwen)
    state = apply_qwen_streaming_patch()
    assert state.available is True
    assert state.patched is False


def test_runtime_unavailable_without_native_stream(monkeypatch):
    class FakeQwen:
        def __init__(self):
            self.model = object()

        def _convert_inputs_2_neural_method_inputs(self, _kwargs, model_type="voice_clone"):
            assert model_type == "voice_clone"
            return {
                "input_ids": np.array([[1]]),
                "attention_mask": np.array([[1]]),
                "speaker_embedding": None,
                "text_tokens_in_input_ids": None,
            }

        def create_voice_clone_prompt(self, **kwargs):
            del kwargs
            return {"ok": True}

    _install_fake_qwen_module(monkeypatch, FakeQwen)
    state = apply_qwen_streaming_patch()
    assert state.available is True

    model = FakeQwen()
    assert qwen_streaming_runtime_available(model) is False
    with pytest.raises(NotImplementedError):
        list(
            model.stream_generate_voice_clone(
                text="hello",
                language="Auto",
                voice_clone_prompt={"x": 1},
            )
        )

