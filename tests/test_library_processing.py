from modules.core_components.library_processing import (
    WordTimestampLike,
    clean_transcription_for_engine,
    language_to_code,
    parse_asr_model,
    split_into_segments,
)


def test_parse_asr_model_variants():
    assert parse_asr_model("Qwen3 ASR - Large") == ("Qwen3 ASR", "Large")
    assert parse_asr_model("VibeVoice ASR - Default") == ("VibeVoice ASR", None)
    assert parse_asr_model("Whisper - Medium") == ("Whisper", "Medium")


def test_language_to_code_handles_auto_and_known_values():
    assert language_to_code("Auto-detect") is None
    assert language_to_code("English") == "en"
    assert language_to_code("Spanish") == "es"
    assert language_to_code("Unknown") is None


def test_clean_transcription_for_vibevoice():
    raw = "[SPEAKER_1]: hello [noise] world"
    cleaned = clean_transcription_for_engine("VibeVoice ASR", raw)
    assert cleaned == "hello world"


def test_split_into_segments_creates_silence_cut_segments():
    words = [
        WordTimestampLike("Hello", 0.0, 0.4),
        WordTimestampLike("world.", 0.45, 0.9),
        WordTimestampLike("This", 3.0, 3.3),
        WordTimestampLike("is", 3.35, 3.6),
        WordTimestampLike("test.", 3.65, 4.0),
    ]
    segments = split_into_segments(
        full_text="Hello world. This is test.",
        word_timestamps=words,
        min_duration=0.2,
        max_duration=10.0,
        silence_trim=1.0,
        discard_under=0.1,
    )

    assert len(segments) >= 2
    for start, end, text in segments:
        assert end > start
        assert text.strip()
