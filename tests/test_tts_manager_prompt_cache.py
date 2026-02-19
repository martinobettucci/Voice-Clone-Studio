import torch

import modules.core_components.ai_models.tts_manager as tts_module
from modules.core_components.ai_models.tts_manager import TTSManager


def test_voice_prompt_memory_cache_has_lru_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_module, "get_device", lambda: "cpu")

    manager = TTSManager(
        user_config={"voice_prompt_memory_cache_limit": 2},
        samples_dir=tmp_path,
    )

    prompt = {"tensor": torch.tensor([1.0], dtype=torch.float32)}

    assert manager.save_voice_prompt("s1", prompt, "h1", model_size="1.7B")
    assert manager.save_voice_prompt("s2", prompt, "h2", model_size="1.7B")
    assert manager.save_voice_prompt("s3", prompt, "h3", model_size="1.7B")

    assert manager.load_voice_prompt("s1", "h1", model_size="1.7B") is not None
    assert manager.load_voice_prompt("s2", "h2", model_size="1.7B") is not None
    assert list(manager._voice_prompt_cache.keys()) == ["s1_1.7B", "s2_1.7B"]

    # Touch s1 so it becomes most-recent; then loading s3 should evict s2.
    assert manager.load_voice_prompt("s1", "h1", model_size="1.7B") is not None
    assert manager.load_voice_prompt("s3", "h3", model_size="1.7B") is not None
    assert list(manager._voice_prompt_cache.keys()) == ["s1_1.7B", "s3_1.7B"]


def test_luxtts_prompt_memory_cache_has_lru_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_module, "get_device", lambda: "cpu")

    manager = TTSManager(
        user_config={"luxtts_prompt_memory_cache_limit": 2},
        samples_dir=tmp_path,
    )

    encoded_prompt = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    assert manager.save_luxtts_prompt("s1", encoded_prompt, "a1", rms=0.01, ref_duration=30)
    assert manager.save_luxtts_prompt("s2", encoded_prompt, "a2", rms=0.01, ref_duration=30)
    assert manager.save_luxtts_prompt("s3", encoded_prompt, "a3", rms=0.01, ref_duration=30)

    assert manager.load_luxtts_prompt("s1", "a1", rms=0.01, ref_duration=30) is not None
    assert manager.load_luxtts_prompt("s2", "a2", rms=0.01, ref_duration=30) is not None
    assert list(manager._luxtts_prompt_cache.keys()) == ["s1", "s2"]

    # Touch s1 so it becomes most-recent; then loading s3 should evict s2.
    assert manager.load_luxtts_prompt("s1", "a1", rms=0.01, ref_duration=30) is not None
    assert manager.load_luxtts_prompt("s3", "a3", rms=0.01, ref_duration=30) is not None
    assert list(manager._luxtts_prompt_cache.keys()) == ["s1", "s3"]
