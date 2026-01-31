"""Core components package for Voice Clone Studio"""

from .confirmation_modal import (
    CONFIRMATION_MODAL_CSS,
    CONFIRMATION_MODAL_HEAD,
    CONFIRMATION_MODAL_HTML,
    show_confirmation_modal_js
)

from .input_modal import (
    INPUT_MODAL_CSS,
    INPUT_MODAL_HEAD,
    INPUT_MODAL_HTML,
    show_input_modal_js
)

from .emotion_manager import (
    CORE_EMOTIONS,
    load_emotions_from_config,
    get_emotion_choices,
    calculate_emotion_values,
    handle_save_emotion,
    handle_delete_emotion
)

__all__ = [
    # Confirmation modal
    "CONFIRMATION_MODAL_CSS",
    "CONFIRMATION_MODAL_HEAD",
    "CONFIRMATION_MODAL_HTML",
    "show_confirmation_modal_js",
    # Input modal
    "INPUT_MODAL_CSS",
    "INPUT_MODAL_HEAD",
    "INPUT_MODAL_HTML",
    "show_input_modal_js",
    # Emotion manager
    "CORE_EMOTIONS",
    "load_emotions_from_config",
    "get_emotion_choices",
    "calculate_emotion_values",
    "handle_save_emotion",
    "handle_delete_emotion",
]
