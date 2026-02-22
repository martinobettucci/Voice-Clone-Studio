from __future__ import annotations

import numpy as np
import torch

import modules.core_components.ai_models.tts_manager as tts_manager_mod
from modules.core_components.ai_models.tts_manager import TTSManager


def test_chatterbox_generate_forwards_and_clamps_expert_params(monkeypatch):
    mgr = TTSManager(user_config={})
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 32), dtype=torch.float32)

    monkeypatch.setattr(tts_manager_mod, "set_seed", lambda _seed: None)
    monkeypatch.setattr(mgr, "get_chatterbox_tts", lambda: FakeModel())

    audio, sr = mgr.generate_voice_clone_chatterbox(
        text="hello",
        voice_sample_path="/tmp/sample.wav",
        min_p=9.9,
        max_new_tokens=99999,
    )

    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["min_p"] == 0.30
    assert captured["max_new_tokens"] == 4096


def test_chatterbox_multilingual_forwards_and_clamps_expert_params(monkeypatch):
    mgr = TTSManager(user_config={})
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 16), dtype=torch.float32)

    monkeypatch.setattr(tts_manager_mod, "set_seed", lambda _seed: None)
    monkeypatch.setattr(mgr, "get_chatterbox_multilingual", lambda: FakeModel())

    audio, sr = mgr.generate_voice_clone_chatterbox_multilingual(
        text="hello",
        language_code="en",
        voice_sample_path="/tmp/sample.wav",
        min_p=-3.0,
        max_new_tokens=1,
    )

    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["min_p"] == 0.0
    assert captured["max_new_tokens"] == 256


def test_chatterbox_vc_forwards_steps_and_auto_none(monkeypatch):
    mgr = TTSManager(user_config={})
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 24), dtype=torch.float32)

    monkeypatch.setattr(mgr, "get_chatterbox_vc", lambda: FakeModel())

    audio, sr = mgr.generate_voice_convert_chatterbox(
        source_audio_path="/tmp/src.wav",
        target_voice_path="/tmp/target.wav",
        n_cfm_timesteps=0,
    )
    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["n_cfm_timesteps"] is None

    captured.clear()
    audio, sr = mgr.generate_voice_convert_chatterbox(
        source_audio_path="/tmp/src.wav",
        target_voice_path="/tmp/target.wav",
        n_cfm_timesteps=999,
    )
    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["n_cfm_timesteps"] == 30

