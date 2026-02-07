"""
AI Models Management Package

Centralized management for TTS and ASR models.
"""

from .model_utils import (
    get_device,
    get_dtype,
    get_attention_implementation,
    check_model_available_locally,
    empty_cuda_cache,
    log_gpu_memory,
    get_trained_models,
    get_trained_model_names,
    train_model,
)

from .tts_manager import (
    TTSManager,
    get_tts_manager,
)

from .asr_manager import (
    ASRManager,
    get_asr_manager,
)

__all__ = [
    # Utilities
    "get_device",
    "get_dtype",
    "get_attention_implementation",
    "check_model_available_locally",
    "empty_cuda_cache",
    "log_gpu_memory",
    "get_trained_models",
    "get_trained_model_names",
    "train_model",

    # TTS
    "TTSManager",
    "get_tts_manager",

    # ASR
    "ASRManager",
    "get_asr_manager",
]
