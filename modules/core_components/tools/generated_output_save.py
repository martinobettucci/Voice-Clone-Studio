"""Helpers for preview-audio save workflows in generation tabs."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any

import soundfile as sf


def parse_modal_submission(value: str | None, context_prefix: str) -> tuple[bool, bool, str | None]:
    """Parse input modal trigger value.

    Returns:
        (matched_context, is_cancel, filename_or_none)
    """
    if not value or not value.startswith(context_prefix):
        return False, False, None

    raw = value[len(context_prefix) :]
    if raw == "cancel" or raw.startswith("cancel_"):
        return True, True, None

    # Optional save-mode payloads look like: "replace::name_<ts>"
    if "::" in raw:
        _mode, raw = raw.split("::", 1)

    # Input modal appends a timestamp suffix: "<name>_<epoch_ms>"
    name_parts = raw.rsplit("_", 1)
    if len(name_parts) == 2 and name_parts[1].isdigit() and len(name_parts[1]) >= 10:
        raw = name_parts[0]

    return True, False, raw


def sanitize_output_name(raw_name: str | None) -> str:
    """Normalize user file name using modal-compatible rules."""
    if raw_name is None:
        return ""
    clean = "".join(c if c.isalnum() or c in "-_ " else "" for c in str(raw_name)).strip()
    return clean.replace(" ", "_")


def get_existing_wav_stems(output_dir: Path) -> list[str]:
    """List existing .wav stems for overwrite confirmation."""
    if not output_dir.exists():
        return []
    return sorted(f.stem for f in output_dir.glob("*.wav"))


def _as_audio_path(audio_value: Any) -> Path | None:
    if isinstance(audio_value, str) and audio_value.strip():
        return Path(audio_value)
    if isinstance(audio_value, dict):
        path_like = audio_value.get("path") or audio_value.get("name")
        if isinstance(path_like, str) and path_like.strip():
            return Path(path_like)
    return None


def persist_audio_to_wav(audio_value: Any, output_path: Path, default_sample_rate: int = 24000) -> None:
    """Persist an audio component value to a WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_path = _as_audio_path(audio_value)
    if source_path is not None:
        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {source_path}")
        if source_path.resolve() != output_path.resolve():
            shutil.copy2(str(source_path), str(output_path))
        return

    # Tuple/list form: (sample_rate, data)
    if isinstance(audio_value, (tuple, list)) and len(audio_value) >= 2:
        sr, data = audio_value[0], audio_value[1]
        if data is None:
            raise ValueError("Audio data is empty.")
        try:
            sample_rate = int(sr)
        except Exception:
            sample_rate = int(default_sample_rate)
        if sample_rate <= 0:
            sample_rate = int(default_sample_rate)
        sf.write(str(output_path), data, sample_rate)
        return

    # Dict form: {"sample_rate": ..., "data": ...}
    if isinstance(audio_value, dict) and "data" in audio_value:
        data = audio_value.get("data")
        sr = audio_value.get("sample_rate", default_sample_rate)
        if data is None:
            raise ValueError("Audio data is empty.")
        sample_rate = int(sr) if sr else int(default_sample_rate)
        sf.write(str(output_path), data, sample_rate)
        return

    raise ValueError("Unsupported audio value type for saving.")


def save_generated_output(
    audio_value: Any,
    output_dir: Path,
    raw_name: str | None,
    metadata_text: str | None = None,
    default_sample_rate: int = 24000,
) -> Path:
    """Save edited/generated audio to output directory and optional metadata."""
    clean_name = sanitize_output_name(raw_name)
    if not clean_name:
        raise ValueError("Invalid filename.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clean_name}.wav"
    persist_audio_to_wav(audio_value, output_path, default_sample_rate=default_sample_rate)

    if metadata_text:
        metadata_out = "\n".join(line.lstrip() for line in str(metadata_text).lstrip().splitlines())
        output_path.with_suffix(".txt").write_text(metadata_out, encoding="utf-8")

    return output_path


def convert_audio_file_to_mp3(source_path: Path, output_path: Path | None = None) -> Path:
    """Convert an existing audio file to MP3 using ffmpeg."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {src}")

    dst = Path(output_path) if output_path else src.with_suffix(".mp3")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(dst),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise FileNotFoundError("ffmpeg not found. Please install ffmpeg.") from e

    if result.returncode != 0 or not dst.exists():
        err = (result.stderr or result.stdout or "Unknown ffmpeg error").strip()
        err_line = err.splitlines()[-1][:200] if err else "Unknown ffmpeg error"
        raise RuntimeError(f"Failed to convert to MP3: {err_line}")

    return dst
