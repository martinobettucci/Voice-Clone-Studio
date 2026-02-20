"""Shared policy helpers for native live-streaming capability checks."""

from __future__ import annotations


LIVE_STREAM_UNAVAILABLE_NOTICE = "Live streaming not available for this mode. Generated final output only."


def prefix_non_stream_status(status: str) -> str:
    """Prefix a status message with the non-streaming notice."""
    status = (status or "").strip()
    if not status:
        return f"INFO: {LIVE_STREAM_UNAVAILABLE_NOTICE}"
    return f"INFO: {LIVE_STREAM_UNAVAILABLE_NOTICE}\n{status}"


def voice_clone_stream_supported(model_selection: str, tts_manager) -> bool:
    """Return whether selected Voice Clone model can live-stream natively."""
    model_selection = (model_selection or "").strip()
    if not model_selection:
        return False

    if "VibeVoice" in model_selection:
        return bool(tts_manager.supports_live_streaming("vibevoice", "voice_clone"))
    if "Qwen3" in model_selection:
        return bool(tts_manager.supports_live_streaming("qwen", "voice_clone"))
    return False


def conversation_stream_supported(model_type: str, tts_manager) -> bool:
    """Return whether selected Conversation model can live-stream natively."""
    model_type = (model_type or "").strip()
    if model_type == "VibeVoice":
        return bool(tts_manager.supports_live_streaming("vibevoice", "conversation"))
    if model_type == "Qwen Base":
        return bool(tts_manager.supports_live_streaming("qwen", "conversation"))
    return False
