"""
ASR Model Manager

Centralized management for all ASR models (Whisper, VibeVoice ASR, etc.)
"""

import torch
from pathlib import Path
from typing import Dict, Optional

import gc

from .model_utils import (
    get_device, get_dtype, get_attention_implementation,
    get_configured_models_dir,
    resolve_model_source,
    empty_device_cache, log_gpu_memory,
    run_pre_load_hooks
)


class ASRManager:
    """Manages all ASR models with lazy loading and VRAM optimization."""

    def __init__(self, user_config: Dict = None):
        """
        Initialize ASR Manager.

        Args:
            user_config: User configuration dict
        """
        self.user_config = user_config or {}

        # Model cache
        self._whisper_model = None
        self._vibevoice_asr_model = None
        self._qwen3_asr_model = None
        self._qwen3_aligner_model = None

        # Availability flags
        self.whisper_available = self._check_whisper_available()
        self.qwen3_asr_available = self._check_qwen3_asr_available()
        self._last_loaded_model = None

    def _check_whisper_available(self) -> bool:
        """Check if Whisper is installed."""
        try:
            import whisper
            return True
        except ImportError:
            return False

    def _check_qwen3_asr_available(self):
        """Check if qwen-asr package is installed."""
        try:
            from qwen_asr import Qwen3ASRModel
            return True
        except ImportError:
            return False

    def _check_and_unload_if_different(self, model_id):
        """If switching to a different model, unload all. Stops external servers on first/new load."""
        if self._last_loaded_model is not None and self._last_loaded_model != model_id:
            print(f"Switching from {self._last_loaded_model} to {model_id} - unloading all ASR models...")
            self.unload_all()
            run_pre_load_hooks()
        elif self._last_loaded_model is None:
            # First model load — stop external servers (e.g., llama.cpp) to free VRAM
            run_pre_load_hooks()
        self._last_loaded_model = model_id

    def _load_model_with_attention(self, model_class, model_name: str, **kwargs):
        """
        Load a HuggingFace model with best available attention mechanism.

        Returns:
            Tuple: (loaded_model, attention_mechanism_used)
        """
        offline_mode = self.user_config.get("offline_mode", False)

        model_to_load = resolve_model_source(
            model_name,
            offline_mode=offline_mode,
            settings_download_name=model_name.split("/")[-1],
            auto_download_when_online=True,
        )

        mechanisms = get_attention_implementation(
            self.user_config.get("attention_mechanism", "auto")
        )

        last_error = None
        for attn in mechanisms:
            try:
                model = model_class.from_pretrained(
                    model_to_load,
                    attn_implementation=attn,
                    trust_remote_code=True,
                    **kwargs
                )
                print(f"✓ Model loaded with {attn}")
                return model, attn
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e

                is_attn_error = any(
                    keyword in error_msg
                    for keyword in ["flash", "attention", "sdpa", "not supported"]
                )

                if is_attn_error:
                    print(f"  {attn} not available, trying next option...")
                    continue
                else:
                    raise e

        raise RuntimeError(f"Failed to load model: {str(last_error)}")

    def get_whisper(self, size="medium"):
        """Load Whisper ASR model.

        Args:
            size: Model size - "medium" or "large" (default: "medium")
        """
        if not self.whisper_available:
            raise ImportError("Whisper is not installed on this system")

        # Map friendly names to whisper model names
        size_map = {"Medium": "medium", "Large": "large", "medium": "medium", "large": "large"}
        whisper_size = size_map.get(size, "medium")

        # Unload if switching sizes
        if self._whisper_model is not None and getattr(self, '_whisper_size', None) != whisper_size:
            print(f"Switching Whisper model from {self._whisper_size} to {whisper_size}...")
            del self._whisper_model
            self._whisper_model = None
            import gc, torch
            gc.collect()
            empty_device_cache()

        self._check_and_unload_if_different("whisper_asr")

        if self._whisper_model is None:
            print(f"Loading Whisper ASR model ({whisper_size})...")
            import whisper
            offline_mode = self.user_config.get("offline_mode", False)

            whisper_cache = get_configured_models_dir() / "whisper"
            whisper_cache.mkdir(parents=True, exist_ok=True)

            model_url = whisper._MODELS.get(whisper_size)
            model_filename = Path(model_url).name if model_url else None
            model_file_path = whisper_cache / model_filename if model_filename else None

            if offline_mode and (model_file_path is None or not model_file_path.exists()):
                display_name = "Whisper-Medium" if whisper_size == "medium" else "Whisper-Large"
                raise RuntimeError(
                    "❌ Offline mode enabled and Whisper model file is missing locally: "
                    f"{model_file_path if model_file_path else whisper_size}\n"
                    "Download it in Settings -> Model Downloading "
                    f"('{display_name}') or disable Offline Mode."
                )

            if offline_mode:
                # Load directly from local checkpoint path to avoid any network fallback.
                self._whisper_model = whisper.load_model(
                    str(model_file_path),
                    download_root=str(whisper_cache)
                )
                alignment_heads = whisper._ALIGNMENT_HEADS.get(whisper_size)
                if alignment_heads is not None:
                    self._whisper_model.set_alignment_heads(alignment_heads)
            else:
                self._whisper_model = whisper.load_model(
                    whisper_size,
                    download_root=str(whisper_cache)
                )

            self._whisper_size = whisper_size
            print("Whisper ASR model loaded!")

        return self._whisper_model

    def get_qwen3_asr(self, size="Small"):
        """Load Qwen3 ASR model.

        Args:
            size: "Small" (0.6B) or "Large" (1.7B)
        """
        if not self.qwen3_asr_available:
            raise ImportError(
                "qwen-asr package is not installed.\n"
                "Install with: pip install -U qwen-asr"
            )

        model_id = f"qwen3_asr_{size}"
        self._check_and_unload_if_different(model_id)

        if self._qwen3_asr_model is None:
            size_map = {"Small": "0.6B", "Large": "1.7B"}
            model_size = size_map.get(size, "0.6B")
            model_name = f"Qwen/Qwen3-ASR-{model_size}"
            print(f"Loading Qwen3 ASR model ({model_size})...")
            from qwen_asr import Qwen3ASRModel
            device = get_device()
            dtype = get_dtype(device)

            offline_mode = self.user_config.get("offline_mode", False)
            model_to_load = resolve_model_source(
                model_name,
                offline_mode=offline_mode,
                settings_download_name=f"Qwen3-ASR-{model_size}",
                auto_download_when_online=True,
            )

            # Try loading with attention fallback
            mechanisms = get_attention_implementation(
                self.user_config.get("attention_mechanism", "auto")
            )

            model = None
            for attn in mechanisms:
                try:
                    kwargs = dict(
                        dtype=dtype,
                        device_map=device,
                        max_inference_batch_size=32,
                        max_new_tokens=512,
                    )
                    # Only pass attn_implementation for non-eager (eager is default)
                    if attn != "eager":
                        kwargs["attn_implementation"] = attn
                    model = Qwen3ASRModel.from_pretrained(model_to_load, **kwargs)
                    print(f"Qwen3 ASR loaded with {attn} attention")
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    is_attn_error = any(
                        keyword in error_msg
                        for keyword in ["flash", "attention", "sdpa", "not supported"]
                    )
                    if is_attn_error and attn != mechanisms[-1]:
                        print(f"  {attn} not available, trying next option...")
                        continue
                    raise e

            # Create wrapper for consistent API (returns {"text": "..."} like Whisper)
            class Qwen3ASRWrapper:
                def __init__(self, model):
                    self.model = model

                def transcribe(self, audio_path, **kwargs):
                    """Transcribe audio file. Returns dict with 'text' and 'language' keys."""
                    import logging
                    language = kwargs.get("language", None)
                    # Suppress "temperature not valid" warning from transformers generate()
                    gen_logger = logging.getLogger("transformers.generation.utils")
                    prev_level = gen_logger.level
                    gen_logger.setLevel(logging.ERROR)
                    try:
                        results = self.model.transcribe(audio=audio_path, language=language)
                    finally:
                        gen_logger.setLevel(prev_level)
                    text = results[0].text if results else ""
                    detected_language = results[0].language if results else None
                    return {"text": text, "language": detected_language}

            self._qwen3_asr_model = Qwen3ASRWrapper(model)
            print("Qwen3 ASR model loaded!")

        return self._qwen3_asr_model

    def get_qwen3_forced_aligner(self):
        """Load Qwen3 ForcedAligner for word-level timestamps.

        Supports up to 5 minutes of audio in 11 languages:
        Chinese, English, Cantonese, French, German, Italian,
        Japanese, Korean, Portuguese, Russian, Spanish.
        """
        if not self.qwen3_asr_available:
            raise ImportError(
                "qwen-asr package is not installed.\n"
                "Install with: pip install -U qwen-asr"
            )

        if self._qwen3_aligner_model is None:
            print("Loading Qwen3 ForcedAligner...")
            from qwen_asr import Qwen3ForcedAligner

            model_name = "Qwen/Qwen3-ForcedAligner-0.6B"
            device = get_device()
            dtype = get_dtype(device)

            offline_mode = self.user_config.get("offline_mode", False)
            model_to_load = resolve_model_source(
                model_name,
                offline_mode=offline_mode,
                settings_download_name="Qwen3-ForcedAligner-0.6B",
                auto_download_when_online=True,
            )

            self._qwen3_aligner_model = Qwen3ForcedAligner.from_pretrained(
                model_to_load,
                dtype=dtype,
                device_map=device,
            )
            print("Qwen3 ForcedAligner loaded!")

        return self._qwen3_aligner_model

    def unload_forced_aligner(self):
        """Unload forced aligner to free VRAM."""
        if self._qwen3_aligner_model is not None:
            del self._qwen3_aligner_model
            self._qwen3_aligner_model = None
            empty_device_cache()
            print("Qwen3 ForcedAligner unloaded")

    def get_vibevoice_asr(self):
        """Load VibeVoice ASR model."""
        self._check_and_unload_if_different("vibevoice_asr")

        if self._vibevoice_asr_model is None:
            print("Loading VibeVoice ASR model...")
            try:
                from modules.vibevoice_asr.modular.modeling_vibevoice_asr import (
                    VibeVoiceASRForConditionalGeneration
                )
                from modules.vibevoice_asr.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
                import warnings
                import logging
                import json

                model_repo_id = "microsoft/VibeVoice-ASR"
                device = get_device()
                dtype = get_dtype(device)
                offline_mode = self.user_config.get("offline_mode", False)

                model_source = resolve_model_source(
                    model_repo_id,
                    offline_mode=offline_mode,
                    settings_download_name="VibeVoice-ASR",
                    auto_download_when_online=True,
                )

                tokenizer_repo_id = "Qwen/Qwen2.5-1.5B"
                tokenizer_source = resolve_model_source(
                    tokenizer_repo_id,
                    offline_mode=offline_mode,
                    settings_download_name="Qwen2.5-1.5B (VibeVoice Tokenizer)",
                    auto_download_when_online=True,
                )

                # Suppress warnings
                prev_level = logging.getLogger("transformers.tokenization_utils_base").level
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    processor = VibeVoiceASRProcessor.from_pretrained(
                        model_source,
                        local_files_only=offline_mode,
                        language_model_pretrained_name=tokenizer_source,
                    )

                logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

                # Load model with configured attention
                mechanisms = get_attention_implementation(
                    self.user_config.get("attention_mechanism", "auto")
                )

                model = None
                for attn in mechanisms:
                    try:
                        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                            model_source,
                            dtype=dtype,
                            device_map=device if device == "auto" else None,
                            attn_implementation=attn,
                            trust_remote_code=True,
                            low_cpu_mem_usage=self.user_config.get("low_cpu_mem_usage", False),
                            local_files_only=offline_mode
                        )
                        print(f"✓ VibeVoice ASR loaded with {attn} attention")
                        break
                    except Exception as e:
                        if attn != mechanisms[-1]:
                            print(f"  {attn} not available, trying next option...")
                            continue
                        raise e

                if device != "auto":
                    model = model.to(device)

                model.eval()

                # Create inference wrapper
                class VibeVoiceWrapper:
                    def __init__(self, model, processor, device):
                        self.model = model
                        self.processor = processor
                        self.device = device

                    def transcribe(self, audio_path):
                        """Transcribe audio file."""
                        # Process audio
                        inputs = self.processor(
                            audio=audio_path,
                            return_tensors="pt",
                            add_generation_prompt=True
                        )

                        # Move to device
                        inputs = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in inputs.items()
                        }

                        # Generate
                        with torch.no_grad():
                            output_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=512,
                                temperature=None,
                                do_sample=False,
                                num_beams=1,
                                pad_token_id=self.processor.pad_id,
                                eos_token_id=self.processor.tokenizer.eos_token_id,
                            )

                        # Decode
                        generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
                        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

                        # Parse output
                        try:
                            segments = self.processor.post_process_transcription(generated_text)
                            formatted_lines = []
                            for segment in segments:
                                speaker = segment.get("Speaker", segment.get("speaker_id", 0))
                                content = segment.get("Content", segment.get("text", "")).strip()
                                if content:
                                    formatted_lines.append(f"[Speaker {speaker}]: {content}")
                            formatted_text = "\n".join(formatted_lines)
                        except:
                            try:
                                json_start = generated_text.find("[")
                                if json_start != -1:
                                    json_text = generated_text[json_start:]
                                    segments = json.loads(json_text)
                                    formatted_lines = []
                                    for segment in segments:
                                        speaker = segment.get("Speaker", 0)
                                        content = segment.get("Content", "").strip()
                                        if content:
                                            formatted_lines.append(f"[Speaker {speaker}]: {content}")
                                    formatted_text = "\n".join(formatted_lines)
                                else:
                                    formatted_text = generated_text
                            except:
                                formatted_text = generated_text

                        return {"text": formatted_text}

                self._vibevoice_asr_model = VibeVoiceWrapper(model, processor, device)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"VibeVoice ASR loaded! ({total_params / 1e9:.2f}B parameters)")

            except ImportError as e:
                print(f"❌ VibeVoice ASR not available: {e}")
                raise
            except Exception as e:
                print(f"❌ Error loading VibeVoice ASR: {e}")
                raise

        return self._vibevoice_asr_model

    def unload_all(self):
        """Unload all ASR models to free VRAM."""
        freed = []

        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            freed.append("Whisper")

        if self._vibevoice_asr_model is not None:
            del self._vibevoice_asr_model
            self._vibevoice_asr_model = None
            freed.append("VibeVoice ASR")

        if self._qwen3_asr_model is not None:
            del self._qwen3_asr_model
            self._qwen3_asr_model = None
            freed.append("Qwen3 ASR")

        if self._qwen3_aligner_model is not None:
            del self._qwen3_aligner_model
            self._qwen3_aligner_model = None
            freed.append("Qwen3 ForcedAligner")

        if freed:
            gc.collect()
            empty_device_cache()
            print(f"Unloaded ASR models: {', '.join(freed)}")

        return bool(freed)


# Global singleton instance
_asr_manager = None


def get_asr_manager(user_config: Dict = None) -> ASRManager:
    """Get or create the global ASR manager."""
    global _asr_manager
    if _asr_manager is None:
        _asr_manager = ASRManager(user_config)
    elif user_config is not None:
        _asr_manager.user_config = user_config
    return _asr_manager
