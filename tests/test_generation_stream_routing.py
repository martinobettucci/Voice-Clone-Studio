from __future__ import annotations

from modules.core_components.tools.live_stream_policy import (
    LIVE_STREAM_UNAVAILABLE_NOTICE,
    conversation_stream_supported,
    prefix_non_stream_status,
    voice_clone_stream_supported,
)


class _FakeTTSManager:
    def supports_live_streaming(self, engine: str, mode: str) -> bool:
        key = (engine, mode)
        return key in {
            ("qwen", "voice_clone"),
            ("qwen", "conversation"),
            ("vibevoice", "voice_clone"),
            ("vibevoice", "conversation"),
        }


def test_voice_clone_stream_routing():
    mgr = _FakeTTSManager()
    assert voice_clone_stream_supported("Qwen3 - Large", mgr) is True
    assert voice_clone_stream_supported("VibeVoice - Large", mgr) is True
    assert voice_clone_stream_supported("LuxTTS - Default", mgr) is False
    assert voice_clone_stream_supported("Chatterbox - Default", mgr) is False


def test_conversation_stream_routing():
    mgr = _FakeTTSManager()
    assert conversation_stream_supported("Qwen Base", mgr) is True
    assert conversation_stream_supported("VibeVoice", mgr) is True
    assert conversation_stream_supported("Qwen Speakers", mgr) is False
    assert conversation_stream_supported("LuxTTS", mgr) is False


def test_non_stream_status_prefix():
    text = "Ready to save"
    out = prefix_non_stream_status(text)
    assert LIVE_STREAM_UNAVAILABLE_NOTICE in out
    assert text in out

