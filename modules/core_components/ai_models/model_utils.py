"""
Model utilities for AI model management.

Shared utilities for model loading, device management, and VRAM optimization.
"""

import json
import os
import random
import torch
import numpy as np
from pathlib import Path


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device=None):
    """Get appropriate dtype based on device.

    CUDA uses bfloat16 for best quality.
    MPS uses float32 (float16 causes torch.multinomial failures during
    sampling due to NaN/subnormal values from softmax precision loss;
    unified memory makes float32 low-cost on Apple Silicon).
    CPU uses float32.
    """
    if device is None:
        device = get_device()

    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


def get_attention_implementation(user_preference="auto"):
    """
    Get list of attention implementations to try, in order.

    Args:
        user_preference: User's attention preference from config:
            - "auto": Try best options in order
            - "flash_attention_2": Use Flash Attention 2
            - "sdpa": Use Scaled Dot-Product Attention
            - "eager": Use eager attention

    Returns:
        List of attention mechanism strings to try
    """
    if user_preference == "flash_attention_2":
        return ["flash_attention_2", "sdpa", "eager"]
    elif user_preference == "sdpa":
        return ["sdpa", "flash_attention_2", "eager"]
    elif user_preference == "eager":
        return ["eager"]
    else:  # "auto"
        return ["flash_attention_2", "sdpa", "eager"]


def get_project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent.parent


def get_configured_models_dir():
    """Get configured models directory from config.json (fallback: <project>/models)."""
    project_root = get_project_root()
    config_path = project_root / "config.json"
    default_models_dir = project_root / "models"

    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            configured = config.get("models_folder", "models")
            models_dir = Path(configured)
            if not models_dir.is_absolute():
                models_dir = project_root / models_dir
            return models_dir
    except Exception:
        pass

    return default_models_dir


def check_model_available_locally(model_name):
    """
    Check if model is available in local models directory.

    Args:
        model_name: Model name/path (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    Returns:
        Path to local model or None if not found
    """
    models_dir = get_configured_models_dir()

    # Try exact model name
    model_path = models_dir / model_name.split("/")[-1]
    if model_path.exists() and (
        list(model_path.glob("*.safetensors"))
        or list(model_path.glob("*.onnx"))
        or list(model_path.glob("*.pt"))
        or list(model_path.glob("*.bin"))
    ):
        return model_path

    return None


def resolve_model_source(
    model_name,
    offline_mode=False,
    settings_download_name=None,
    auto_download_when_online=True,
):
    """
    Resolve a model source for runtime loading.

    Priority:
    1) Local cloned model in configured models folder
    2) In offline mode: fail with actionable error
    3) In online mode: attempt local clone download to configured models folder
    4) If local clone download fails, fall back to Hub repo id

    Returns:
        str: local path or repo id
    """
    local_path = check_model_available_locally(model_name)
    if local_path:
        return str(local_path)

    if offline_mode:
        hint = settings_download_name or model_name.split("/")[-1]
        raise RuntimeError(
            "❌ Offline mode enabled and model is missing locally: "
            f"{model_name}\n"
            "Download it in Settings -> Model Downloading "
            f"('{hint}') or disable Offline Mode."
        )

    if auto_download_when_online:
        success, message, path = download_model_from_huggingface(model_name)
        if success and path:
            print(f"Downloaded model for local offline reuse: {model_name} -> {path}")
            return path
        print(
            "Could not pre-download model to local models folder; "
            f"falling back to online hub loading: {model_name}\nReason: {message}"
        )

    return model_name


def download_model_from_huggingface(model_id, models_dir=None, local_folder_name=None, progress=None):
    """Download model from HuggingFace using git clone (not cache).

    Uses git-lfs to download directly to models/ folder without using HF cache.
    Users can also manually clone with:
    git clone https://huggingface.co/{model_id} models/{folder_name}

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        models_dir: Path to models directory (defaults to project models folder)
        local_folder_name: Custom local folder name (default: extract from model_id)
        progress: Optional Gradio progress callback

    Returns:
        Tuple: (success: bool, message: str, local_path: str or None)
    """
    import subprocess
    import threading

    try:
        # Validate inputs
        if not model_id or "/" not in model_id:
            return False, f"Invalid model ID: {model_id}. Use format 'Author/ModelName'", None

        # Determine local folder name
        if not local_folder_name:
            local_folder_name = model_id.split("/")[-1]

        if models_dir is None:
            models_dir = get_configured_models_dir()
        else:
            models_dir = Path(models_dir)

        models_dir.mkdir(exist_ok=True)
        local_path = models_dir / local_folder_name

        # Check if already downloaded (look for model files)
        if (
            list(local_path.glob("*.safetensors"))
            or list(local_path.glob("*.onnx"))
            or list(local_path.glob("*.pt"))
            or list(local_path.glob("*.bin"))
        ):
            return True, f"Model already exists at {local_path}", str(local_path)

        # Check if git-lfs is installed
        try:
            subprocess.run(["git", "lfs", "version"], capture_output=True, check=True, timeout=5)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            error_msg = (
                "git-lfs is not installed or not in PATH. Install from: https://git-lfs.com\n"
                "Or manually download from HuggingFace and place in: models/" + local_folder_name
            )
            print(error_msg, flush=True)
            return False, error_msg, None

        # Clone repository with git-lfs
        hf_url = f"https://huggingface.co/{model_id}"

        try:
            print(f"\nStarting download: {model_id}", flush=True)
            print(f"URL: {hf_url}", flush=True)
            print(f"Destination: {local_path}\n", flush=True)

            # Track download state
            download_complete = {"done": False, "returncode": None}

            def run_download():
                """Run git clone without capturing output so it shows in console."""
                try:
                    result = subprocess.run(
                        ["git", "clone", hf_url, str(local_path)],
                        timeout=3600
                    )
                    download_complete["returncode"] = result.returncode
                except Exception as e:
                    print(f"Download error: {e}", flush=True)
                    download_complete["returncode"] = -1
                finally:
                    download_complete["done"] = True

            # Start download thread
            download_thread = threading.Thread(target=run_download, daemon=True)
            download_thread.start()

            # Wait for download to complete (progress shown in console)
            download_thread.join()

            if download_complete["returncode"] != 0:
                return False, "Download failed. Check console for details.", None

            # Verify model files exist
            if not (
                list(local_path.glob("*.safetensors"))
                or list(local_path.glob("*.onnx"))
                or list(local_path.glob("*.pt"))
                or list(local_path.glob("*.bin"))
            ):
                return False, "Model files not found - download may be incomplete.", None

            print(f"\nSuccessfully downloaded to {local_path}\n", flush=True)
            return True, f"Successfully downloaded to {local_path}", str(local_path)

        except subprocess.TimeoutExpired:
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)
            return False, "Download timed out after 1 hour. Check your internet connection and try again.", None
        except Exception as e:
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path, ignore_errors=True)
            return False, f"Download error: {str(e)}", None

    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None


def empty_device_cache():
    """Empty GPU cache (CUDA or MPS) if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# Keep old name as alias for compatibility
empty_cuda_cache = empty_device_cache


# ============================================================================
# Pre-model-load hooks
# External processes (e.g., llama.cpp server) register shutdown callbacks here
# so they get stopped before any AI model loads to free VRAM.
# ============================================================================

_pre_load_hooks = []


def register_pre_load_hook(hook):
    """Register a callback to run before any AI model is loaded.

    Used by external processes (e.g., llama.cpp) that need to be shut
    down to free VRAM before loading GPU-resident models.
    """
    if hook not in _pre_load_hooks:
        _pre_load_hooks.append(hook)


def run_pre_load_hooks():
    """Run all registered pre-load hooks (e.g., stop llama.cpp server)."""
    for hook in _pre_load_hooks:
        try:
            hook()
        except Exception:
            pass


_ORIG_CUDNN_BENCHMARK = torch.backends.cudnn.benchmark
_ORIG_CUDNN_DETERMINISTIC = torch.backends.cudnn.deterministic
_ORIG_CUDNN_ALLOW_TF32 = torch.backends.cudnn.allow_tf32 if hasattr(torch.backends.cudnn, "allow_tf32") else None
_ORIG_CUDA_MATMUL_ALLOW_TF32 = (
    torch.backends.cuda.matmul.allow_tf32
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul")
    else None
)
_ORIG_DETERMINISTIC_ALGOS = torch.are_deterministic_algorithms_enabled()


def _env_flag(name, default=False):
    """Parse a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def configure_runtime_reproducibility(deterministic_mode=False):
    """Configure backend flags for deterministic vs fast execution.

    Notes:
    - This improves repeatability but cannot guarantee bit-identical output
      for all kernels/models/devices.
    - CUBLAS_WORKSPACE_CONFIG is most effective when set before CUDA context creation.
    """
    deterministic_mode = bool(deterministic_mode)

    if deterministic_mode:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = _ORIG_CUDNN_BENCHMARK
        torch.backends.cudnn.deterministic = _ORIG_CUDNN_DETERMINISTIC
        if _ORIG_CUDNN_ALLOW_TF32 is not None:
            torch.backends.cudnn.allow_tf32 = _ORIG_CUDNN_ALLOW_TF32
        if _ORIG_CUDA_MATMUL_ALLOW_TF32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = _ORIG_CUDA_MATMUL_ALLOW_TF32
        torch.use_deterministic_algorithms(_ORIG_DETERMINISTIC_ALGOS, warn_only=True)

    os.environ["VCS_DETERMINISTIC_MODE"] = "1" if deterministic_mode else "0"


def set_seed(seed, deterministic_mode=None):
    """Set random seed across Python/NumPy/PyTorch and configure determinism."""
    if deterministic_mode is None:
        deterministic_mode = _env_flag("VCS_DETERMINISTIC_MODE", default=False)

    configure_runtime_reproducibility(deterministic_mode)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_gpu_memory(label=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't expose detailed memory stats, just note it's active
        label_str = f" ({label})" if label else ""
        print(f"GPU Memory{label_str}: MPS device active (detailed stats not available)")


def get_trained_models(models_dir=None):
    """
    Find trained model checkpoints in the models directory.

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of dicts with display_name, path, and speaker_name
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    models = []
    if models_dir.exists():
        for folder in models_dir.iterdir():
            if folder.is_dir():
                for checkpoint in folder.glob("checkpoint-*"):
                    if checkpoint.is_dir():
                        models.append({
                            'display_name': f"{folder.name} - {checkpoint.name}",
                            'path': str(checkpoint),
                            'speaker_name': folder.name
                        })
    return models


def get_trained_model_names(models_dir=None):
    """Get list of existing trained model folder names.

    Args:
        models_dir: Path to models directory (defaults to project models folder)

    Returns:
        List of folder name strings
    """
    if models_dir is None:
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    if not models_dir.exists():
        return []

    return [folder.name for folder in models_dir.iterdir() if folder.is_dir()]


def train_model(folder, speaker_name, ref_audio_filename, batch_size,
                learning_rate, num_epochs, save_interval,
                user_config, datasets_dir, project_root,
                play_completion_beep=None, progress=None, trained_models_root=None):
    """Complete training workflow: validate, prepare data, and train model.

    Args:
        folder: Dataset subfolder name
        speaker_name: Name for the trained model/speaker
        ref_audio_filename: Reference audio file from dataset
        batch_size: Training batch size
        learning_rate: Training learning rate
        num_epochs: Number of training epochs
        save_interval: Save checkpoint every N epochs
        user_config: User configuration dict
        datasets_dir: Path to datasets directory
        project_root: Path to project root
        play_completion_beep: Optional callback for audio notification
        progress: Optional Gradio progress callback
    """
    import subprocess
    import sys
    import json
    import os
    from modules.core_components.audio_utils import check_audio_format

    if progress is None:
        def progress(*a, **kw):
            pass

    # ============== STEP 1: Validation ==============
    progress(0.0, desc="Step 1/3: Validating dataset...")

    if not folder or folder == "(No folders)" or folder == "(Select Dataset)":
        return "Error: Please select a dataset folder"

    if not speaker_name or not speaker_name.strip():
        return "Error: Please enter a speaker name"

    if not ref_audio_filename:
        return "Error: Please select a reference audio file"

    if save_interval is None:
        save_interval = 5

    # Create output directory
    if trained_models_root is not None:
        output_dir = Path(trained_models_root) / speaker_name.strip()
    else:
        trained_models_folder = user_config.get("trained_models_folder", "models")
        output_dir = project_root / trained_models_folder / speaker_name.strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = datasets_dir / folder
    if not base_dir.exists():
        return f"Error: Folder not found: {folder}"

    ref_audio_path = base_dir / ref_audio_filename
    if not ref_audio_path.exists():
        return f"Error: Reference audio not found: {ref_audio_filename}"

    # Only get audio files
    audio_files = [f for f in (list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))
                   if not f.name.endswith('.txt') and not f.name.endswith('.jsonl')]
    if not audio_files:
        return "Error: No audio files found in folder"

    issues = []
    valid_files = []
    converted_count = 0
    total = len(audio_files)

    status_log = []
    status_log.append("=" * 60)
    status_log.append("STEP 1/3: DATASET VALIDATION")
    status_log.append("=" * 60)

    for i, audio_path in enumerate(audio_files):
        progress(0.0 + (0.2 * (i + 1) / total), desc=f"Validating {audio_path.name}...")

        txt_path = audio_path.with_suffix(".txt")

        if not txt_path.exists():
            issues.append(f"[X] {audio_path.name}: Missing transcript")
            continue

        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if not transcript:
                issues.append(f"[X] {audio_path.name}: Empty transcript")
                continue
        except Exception:
            issues.append(f"[X] {audio_path.name}: Cannot read transcript")
            continue

        is_correct, info = check_audio_format(str(audio_path))
        if not is_correct:
            if not info:
                issues.append(f"[X] {audio_path.name}: Cannot read audio file")
                continue

            progress(0.0 + (0.2 * (i + 1) / total), desc=f"Converting {audio_path.name}...")
            temp_output = audio_path.parent / f"temp_{audio_path.name}"
            cmd = [
                'ffmpeg', '-y', '-i', str(audio_path),
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
                '-acodec', 'pcm_s16le', str(temp_output)
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and temp_output.exists():
                    audio_path.unlink()
                    temp_output.rename(audio_path)
                    converted_count += 1
                else:
                    issues.append(f"[X] {audio_path.name}: Conversion failed - {result.stderr[:100]}")
                    continue
            except FileNotFoundError:
                issues.append(f"[X] {audio_path.name}: ffmpeg not found")
                continue
            except Exception as e:
                issues.append(f"[X] {audio_path.name}: Conversion error - {str(e)[:100]}")
                continue

        valid_files.append(audio_path.name)

    if not valid_files:
        return "Error: No valid training samples found\n" + "\n".join(issues[:10])

    status_log.append(f"Found {len(valid_files)} valid training samples")
    if converted_count > 0:
        status_log.append(f"Auto-converted {converted_count} files to 24kHz 16-bit mono")
    if issues:
        status_log.append(f"{len(issues)} files skipped:")
        for issue in issues[:5]:
            status_log.append(f"   {issue}")
        if len(issues) > 5:
            status_log.append(f"   ... and {len(issues) - 5} more")

    # Ensure reference audio is correct format
    progress(0.2, desc="Preparing reference audio...")
    is_correct, info = check_audio_format(str(ref_audio_path))
    if not is_correct:
        temp_output = ref_audio_path.parent / f"temp_{ref_audio_path.name}"
        cmd = [
            'ffmpeg', '-y', '-i', str(ref_audio_path),
            '-ar', '24000', '-ac', '1', '-sample_fmt', 's16',
            '-acodec', 'pcm_s16le', str(temp_output)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and temp_output.exists():
            ref_audio_path.unlink()
            temp_output.rename(ref_audio_path)
        else:
            return f"Error: Failed to convert reference audio: {result.stderr[:200]}"

    # Generate train_raw.jsonl
    progress(0.25, desc="Generating train_raw.jsonl...")
    train_raw_path = base_dir / "train_raw.jsonl"
    jsonl_entries = []

    for filename in valid_files:
        audio_path_entry = base_dir / filename
        txt_path = audio_path_entry.with_suffix(".txt")
        transcript = txt_path.read_text(encoding="utf-8").strip()

        entry = {
            "audio": str(audio_path_entry.absolute()),
            "text": transcript,
            "ref_audio": str(ref_audio_path.absolute())
        }
        jsonl_entries.append(entry)

    try:
        with open(train_raw_path, 'w', encoding='utf-8') as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        status_log.append(f"Generated train_raw.jsonl with {len(jsonl_entries)} entries")
    except Exception as e:
        return f"Error: Failed to write train_raw.jsonl: {str(e)}"

    # ============== STEP 2: Prepare Data (extract audio codes) ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 2/3: EXTRACTING AUDIO CODES")
    status_log.append("=" * 60)
    progress(0.3, desc="Step 2/3: Extracting audio codes...")

    train_with_codes_path = base_dir / "train_with_codes.jsonl"
    modules_dir = project_root / "modules"
    prepare_script = modules_dir / "qwen_finetune" / "prepare_data.py"

    if not prepare_script.exists():
        status_log.append("[X] Qwen3-TTS finetuning scripts not found!")
        status_log.append("   Please ensure Qwen3-TTS repository is cloned.")
        return "\n".join(status_log)

    tokenizer_model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    offline_mode = user_config.get("offline_mode", False)
    try:
        tokenizer_model_path = resolve_model_source(
            tokenizer_model_id,
            offline_mode=offline_mode,
            settings_download_name="Qwen3-TTS-Tokenizer-12Hz",
            auto_download_when_online=True,
        )
        if tokenizer_model_path != tokenizer_model_id:
            status_log.append(f"[OK] Using local tokenizer model at: {tokenizer_model_path}")
        else:
            status_log.append(f"[OK] Using tokenizer model from hub: {tokenizer_model_id}")
    except Exception as e:
        status_log.append(f"[X] {str(e)}")
        return "\n".join(status_log)

    prepare_cmd = [
        sys.executable,
        "-u",
        str(prepare_script.absolute()),
        "--device", get_device(),
        "--tokenizer_model_path", tokenizer_model_path,
        "--input_jsonl", str(train_raw_path),
        "--output_jsonl", str(train_with_codes_path)
    ]

    status_log.append(f"Running: {' '.join(prepare_cmd)}")
    status_log.append("")

    try:
        result = subprocess.Popen(
            prepare_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(base_dir)
        )

        for line in result.stdout:
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

        result.wait()

        if result.returncode != 0:
            status_log.append(f"[X] prepare_data.py failed with exit code {result.returncode}")
            return "\n".join(status_log)

        if not train_with_codes_path.exists():
            status_log.append("[X] train_with_codes.jsonl was not generated")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("[OK] Audio codes extracted successfully")

    except Exception as e:
        status_log.append(f"[X] Error running prepare_data.py: {str(e)}")
        return "\n".join(status_log)

    # ============== STEP 3: Fine-tune ==============
    status_log.append("")
    status_log.append("=" * 60)
    status_log.append("STEP 3/3: TRAINING MODEL")
    status_log.append("=" * 60)
    progress(0.5, desc="Step 3/3: Training model (this may take a while)...")

    sft_script = modules_dir / "qwen_finetune" / "sft_12hz.py"

    if not sft_script.exists():
        status_log.append("[X] sft_12hz.py not found!")
        return "\n".join(status_log)

    base_model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    status_log.append(f"Locating base model: {base_model_id}")
    try:
        resolved_base_source = resolve_model_source(
            base_model_id,
            offline_mode=offline_mode,
            settings_download_name="Qwen3-TTS-12Hz-1.7B-Base",
            auto_download_when_online=True,
        )
    except Exception as e:
        status_log.append(f"[X] {str(e)}")
        return "\n".join(status_log)

    if resolved_base_source != base_model_id:
        base_model_path = resolved_base_source
        status_log.append(f"[OK] Using local base model at: {base_model_path}")
    else:
        try:
            from huggingface_hub import snapshot_download
            base_model_path = snapshot_download(
                repo_id=base_model_id,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.npz"],
                local_files_only=False
            )
            status_log.append(f"[OK] Using cached model at: {base_model_path}")
        except Exception as e:
            status_log.append(f"[X] Failed to locate/download base model: {str(e)}")
            return "\n".join(status_log)

    attn_impl = user_config.get("attention_mechanism", "auto")

    sft_cmd = [
        sys.executable,
        "-u",
        str(sft_script.absolute()),
        "--init_model_path", base_model_path,
        "--output_model_path", str(output_dir),
        "--train_jsonl", str(train_with_codes_path),
        "--batch_size", str(int(batch_size)),
        "--lr", str(learning_rate),
        "--num_epochs", str(int(num_epochs)),
        "--save_interval", str(int(save_interval)),
        "--speaker_name", speaker_name.strip().lower(),
        "--attn_implementation", attn_impl
    ]

    status_log.append("")
    status_log.append("Training configuration:")
    status_log.append(f"  Base model: {base_model_id}")
    status_log.append(f"  Attention implementation: {attn_impl}")
    status_log.append(f"  Batch size: {int(batch_size)}")
    status_log.append(f"  Learning rate: {learning_rate}")
    status_log.append(f"  Epochs: {int(num_epochs)}")
    status_log.append(f"  Save interval: Every {int(save_interval)} epoch(s)" if save_interval > 0 else "  Save interval: Every epoch")
    status_log.append(f"  Speaker name: {speaker_name.strip()}")
    status_log.append(f"  Output: {output_dir}")
    status_log.append("")
    status_log.append("Starting training...")
    status_log.append(f"Running: {' '.join([str(arg) for arg in sft_cmd])}")
    status_log.append("")

    try:
        env = os.environ.copy()
        env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        env['TOKENIZERS_PARALLELISM'] = 'false'

        result = subprocess.Popen(
            sft_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        for line in result.stdout:
            line = line.strip()
            if line:
                status_log.append(f"  {line}")

                if "Epoch" in line and "Step" in line:
                    try:
                        epoch_num = int(line.split("Epoch")[1].split("|")[0].strip())
                        progress_val = 0.5 + (0.5 * (epoch_num + 1) / int(num_epochs))
                        progress(progress_val, desc=f"Training: {line[:60]}...")
                    except Exception:
                        pass

        result.wait()

        if result.returncode != 0:
            status_log.append("")
            status_log.append(f"[X] Training failed with exit code {result.returncode}")
            return "\n".join(status_log)

        status_log.append("")
        status_log.append("=" * 60)
        status_log.append("TRAINING COMPLETED SUCCESSFULLY!")
        status_log.append("=" * 60)
        status_log.append(f"Model saved to: {output_dir}")
        status_log.append(f"Speaker name: {speaker_name.strip()}")
        status_log.append("")
        status_log.append("To use your trained model:")
        status_log.append("  1. Go to Voice Presets tab")
        status_log.append("  2. Select 'Trained Models' radio button")
        status_log.append(f"  3. Click refresh and select '{speaker_name.strip()}'")

        progress(1.0, desc="Training complete!")
        if play_completion_beep:
            play_completion_beep()

    except Exception as e:
        status_log.append(f"[X] Error during training: {str(e)}")
        return "\n".join(status_log)

    return "\n".join(status_log)
