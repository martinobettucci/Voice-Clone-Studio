"""
ASR Model Manager

Centralized management for all ASR models (Whisper, VibeVoice ASR, etc.)
"""

import torch
import threading
from pathlib import Path
from typing import Dict, Optional
from functools import wraps

import gc
import types

from .model_utils import (
    get_device, get_dtype, get_attention_implementation,
    get_configured_models_dir,
    resolve_model_source,
    empty_device_cache, log_gpu_memory,
    run_pre_load_hooks
)


def _with_manager_lock(func):
    """Serialize model load/unload mutations within one manager instance."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._manager_lock:
            return func(self, *args, **kwargs)
    return wrapper


def _is_oom_error(exc: Exception) -> bool:
    """Return True if exception looks like an out-of-memory failure."""
    msg = str(exc).lower()
    return "out of memory" in msg


def _patch_qwen3_asr_low_memory_runtime(
    qwen_model,
    *,
    aggressive_cleanup: bool = True,
    debug_memory: bool = False,
):
    """Patch qwen-asr transformers inference to reduce transient VRAM retention."""
    if getattr(qwen_model, "_vcs_low_memory_patch", False):
        return

    if not hasattr(qwen_model, "_infer_asr_transformers"):
        return

    def _infer_asr_transformers_low_mem(self, contexts, wavs, languages):
        outs = []
        texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]

        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 1:
            batch_size = len(texts)

        for i in range(0, len(texts), batch_size):
            sub_text = texts[i : i + batch_size]
            sub_wavs = wavs[i : i + batch_size]

            raw_inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
            model_inputs = {}
            for key, value in raw_inputs.items():
                if isinstance(value, torch.Tensor):
                    tensor = value.to(self.model.device)
                    if torch.is_floating_point(tensor):
                        tensor = tensor.to(self.model.dtype)
                    model_inputs[key] = tensor
                else:
                    model_inputs[key] = value

            generated = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens)
            sequences = generated.sequences if hasattr(generated, "sequences") else generated

            # Drop large generate() internals as soon as possible to avoid retention.
            for attr in ("past_key_values", "logits", "hidden_states", "attentions", "scores"):
                if hasattr(generated, attr):
                    setattr(generated, attr, None)

            prompt_len = model_inputs["input_ids"].shape[1]
            decoded = self.processor.batch_decode(
                sequences[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(list(decoded))

            del decoded
            del sequences
            del generated
            del model_inputs
            del raw_inputs

            if aggressive_cleanup:
                gc.collect()
                empty_device_cache()
                if debug_memory:
                    log_gpu_memory("Qwen3 ASR sub-batch cleanup")

        return outs

    qwen_model._infer_asr_transformers = types.MethodType(_infer_asr_transformers_low_mem, qwen_model)
    qwen_model._vcs_low_memory_patch = True


class _Qwen3ASRWrapper:
    """Thin compatibility wrapper with OOM retry and aggressive cache cleanup."""

    def __init__(
        self,
        model,
        *,
        max_inference_batch_size: int,
        aggressive_cleanup: bool,
        oom_retry: bool,
        debug_memory: bool = False,
    ):
        self.model = model
        self.max_inference_batch_size = max(1, int(max_inference_batch_size))
        self.aggressive_cleanup = bool(aggressive_cleanup)
        self.oom_retry = bool(oom_retry)
        self.debug_memory = bool(debug_memory)

    def _compact_device_cache(self, force_gc: bool = False):
        if force_gc:
            gc.collect()
        empty_device_cache()

    def _clear_generation_state(self):
        core_model = getattr(self.model, "model", None)
        if core_model is None:
            return
        for obj in (core_model, getattr(core_model, "thinker", None)):
            if obj is None:
                continue
            if hasattr(obj, "_cache"):
                obj._cache = None
            if hasattr(obj, "rope_deltas"):
                obj.rope_deltas = None

    def transcribe(self, audio_path, **kwargs):
        """Transcribe audio file. Returns dict with 'text' and 'language' keys."""
        import logging

        language = kwargs.get("language", None)

        # Suppress "temperature not valid" warning from transformers generate()
        gen_logger = logging.getLogger("transformers.generation.utils")
        prev_level = gen_logger.level
        gen_logger.setLevel(logging.ERROR)

        original_batch_size = max(
            1,
            int(getattr(self.model, "max_inference_batch_size", self.max_inference_batch_size)),
        )
        self.model.max_inference_batch_size = original_batch_size

        if self.debug_memory:
            log_gpu_memory("Qwen3 ASR before transcribe")

        results = None
        try:
            try:
                results = self.model.transcribe(audio=audio_path, language=language)
            except Exception as e:
                should_retry = (
                    self.oom_retry
                    and torch.cuda.is_available()
                    and _is_oom_error(e)
                    and original_batch_size > 1
                )
                if not should_retry:
                    raise

                retry_batch_size = max(1, original_batch_size // 2)
                print(
                    "Qwen3 ASR hit OOM. Retrying with reduced inference batch size "
                    f"({original_batch_size} -> {retry_batch_size})..."
                )
                self._clear_generation_state()
                self._compact_device_cache(force_gc=True)
                self.model.max_inference_batch_size = retry_batch_size
                results = self.model.transcribe(audio=audio_path, language=language)
        finally:
            self.model.max_inference_batch_size = original_batch_size
            self._clear_generation_state()
            if self.aggressive_cleanup:
                self._compact_device_cache(force_gc=True)
            if self.debug_memory:
                log_gpu_memory("Qwen3 ASR after transcribe")
            gen_logger.setLevel(prev_level)

        text = results[0].text if results else ""
        detected_language = results[0].language if results else None
        return {"text": text, "language": detected_language}


class ASRManager:
    """Manages all ASR models with lazy loading and VRAM optimization."""

    def __init__(self, user_config: Dict = None):
        """
        Initialize ASR Manager.

        Args:
            user_config: User configuration dict
        """
        self.user_config = user_config or {}
        self._manager_lock = threading.RLock()

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

    @_with_manager_lock
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

    @_with_manager_lock
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
            max_inference_batch_size = max(
                1,
                int(self.user_config.get("qwen_asr_max_inference_batch_size", 8)),
            )
            aggressive_cleanup = bool(self.user_config.get("qwen_asr_aggressive_cleanup", True))
            oom_retry = bool(self.user_config.get("qwen_asr_oom_retry", True))
            debug_memory = bool(self.user_config.get("qwen_asr_debug_memory", False))

            offline_mode = self.user_config.get("offline_mode", False)
            model_to_load = resolve_model_source(
                model_name,
                offline_mode=offline_mode,
                settings_download_name=f"Qwen3-ASR-{model_size}",
                auto_download_when_online=True,
            )

            if debug_memory:
                log_gpu_memory("Qwen3 ASR before load")

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
                        max_inference_batch_size=max_inference_batch_size,
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

            _patch_qwen3_asr_low_memory_runtime(
                model,
                aggressive_cleanup=aggressive_cleanup,
                debug_memory=debug_memory,
            )
            self._qwen3_asr_model = _Qwen3ASRWrapper(
                model,
                max_inference_batch_size=max_inference_batch_size,
                aggressive_cleanup=aggressive_cleanup,
                oom_retry=oom_retry,
                debug_memory=debug_memory,
            )
            if debug_memory:
                log_gpu_memory("Qwen3 ASR after load")
            print("Qwen3 ASR model loaded!")

        return self._qwen3_asr_model

    @_with_manager_lock
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

    @_with_manager_lock
    def unload_forced_aligner(self):
        """Unload forced aligner to free VRAM."""
        if self._qwen3_aligner_model is not None:
            del self._qwen3_aligner_model
            self._qwen3_aligner_model = None
            empty_device_cache()
            print("Qwen3 ForcedAligner unloaded")

    @_with_manager_lock
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

    @_with_manager_lock
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
