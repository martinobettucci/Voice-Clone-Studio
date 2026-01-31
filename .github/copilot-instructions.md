# Voice Clone Studio - Development Guidelines

## CRITICAL: Python Environment

**ALWAYS USE THE PROJECT'S VENV - NO EXCEPTIONS!**

- **NEVER use system Python or bare `python` command**
- **ALWAYS use**: `.\venv\Scripts\python.exe` (Windows) or `./venv/bin/python` (Linux/Mac)
- **Before ANY Python operation**: Use venv Python explicitly
- **Examples**:
  - `.\venv\Scripts\python.exe script.py`
  - `.\venv\Scripts\python.exe -c "import module"`
  - `python script.py` (WRONG - uses system Python)
  - `python -c "import module"` (WRONG - uses system Python)

## Core Architecture Principle

**CRITICAL**: `voice_clone_studio.py` is becoming too large (6000+ lines). **All new features MUST be modularized.**

## Module-First Development

### When Adding ANY New Feature:

#### CREATE A MODULE (Required)
- **Core Components (our code)** → `modules/core_components/[feature].py`
  - Emotion management, UI helpers, configuration, notifications, etc.
  - Anything built specifically for this app's general use
- **New UI Tab** → `modules/core_components/tab_[name].py`
- **Audio Processing** → `modules/core_components/audio_processing/[feature].py`
- **External Code** → `modules/[repo_name]/`
  - Third-party integrations (qwen_finetune, vibevoice_asr, etc.)
  - Keep external repos directly under modules/

#### DON'T ADD TO MAIN FILE
- No new 100+ line functions in `voice_clone_studio.py`
- No duplicating existing functionality
- No business logic in the main UI file

### Module Organization Rules
**`modules/core_components/`** - Our code for general app use:
- emotion_manager.py
- ui_help.py
- confirmation_modal.py
- notification.wav
- Any new core functionality

**`modules/[external_name]/`** - External/third-party code:
- qwen_finetune/ (training logic)
- vibevoice_tts/ (VibeVoice TTS)
- vibevoice_asr/ (VibeVoice ASR)
- Keep external repos separate from our core code

### Existing Module Structure
```
modules/
├── core_components/          # OUR core app components
│   ├── emotion_manager.py    # Emotion system
│   ├── ui_help.py            # Help documentation
│   ├── confirmation_modal.py # Confirmation modal dialog
│   └── notification.wav      # Audio notification
│
├── qwen_finetune/            # EXTERNAL: Training logic
├── vibevoice_tts/            # EXTERNAL: VibeVoice TTS
├── vibevoice_asr/            # EXTERNAL: VibeVoice ASR
└── [new External modules here]
```

### Main File Responsibilities ONLY
- Gradio UI layout structure
- Event handler wiring (`.click()`, `.change()`)
- Configuration loading/saving
- Module imports and orchestration

### Module Design Pattern

**Example: Adding a "Voice Mixer" Feature**

**WRONG** (500 lines in main file):
```python
# voice_clone_studio.py - DON'T DO THIS
with gr.TabItem("Voice Mixer"):
    # 500 lines of UI + logic here
    def complex_mixing_algorithm():
        # 200 lines...
```

**CORRECT** (modular):
```python
# modules/audio_processing/voice_mixer.py
def mix_voices(voice_a, voice_b, ratio):
    """Mix two voice samples."""
    # Implementation here
    return mixed_audio

def create_mixer_ui():
    """Create Voice Mixer tab UI."""
    with gr.TabItem("Voice Mixer"):
        # UI components
        mix_btn = gr.Button("Mix")
        # ...
    return {'button': mix_btn, 'output': output}

# voice_clone_studio.py - MAIN FILE
from modules.audio_processing.voice_mixer import create_mixer_ui, mix_voices

def create_ui():
    # ...
    mixer_ui = create_mixer_ui()
    mixer_ui['button'].click(mix_voices, ...)
```

## Configuration System

### User Preferences (`config.json`)
```python
# Always use config for paths
_user_config.get("samples_folder", "samples")
_user_config.get("models_folder", "models")
_user_config.get("trained_models_folder", "models")
```

### Config Keys:
- `samples_folder` - Voice samples
- `output_folder` - Generated audio
- `datasets_folder` - Training data
- `models_folder` - Downloaded models
- `trained_models_folder` - User-trained models
- `browser_notifications` - Audio notification toggle
- `offline_mode` - Use local models only
- `low_cpu_mem_usage` - Memory optimization
- `attention_mechanism` - flash_attention_2/sdpa/eager

## Model Management Pattern

```python
# Always check before loading
check_and_unload_if_different(model_id)

# Use attention helper
model, attn = load_model_with_attention(
    ModelClass,
    model_name,
    user_preference=_user_config.get("attention_mechanism", "auto"),
    device_map="cuda:0",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=_user_config.get("low_cpu_mem_usage", False)
)

# Respect offline mode
if _user_config.get("offline_mode", False):
    # Check local models only
```

## Emotion System

- **40+ emotions** stored in `config.json` under `"emotions"` key
- **Hardcoded defaults** in `modules/emotion_manager.py` as `CORE_EMOTIONS`
- **Active emotions** loaded into `_active_emotions` global at runtime
- **Intensity**: 0.0-2.0 multiplier
- **Detection**: Regex `\(emotion\)` for Qwen Base conversations
- **Application**: `apply_emotion_preset(emotion, intensity)` adjusts temp/penalty/top_p
- **Management**: Users can save/delete/reset emotions via UI buttons
- **Storage**: Emotions sorted alphabetically (case-insensitive) in config

## Audio Standards

- **Sample Rate**: 24kHz
- **Format**: WAV, 16-bit PCM, Mono
- **Library**: `soundfile` (already installed)
- **Validation**: Use `check_audio_format()` helper

## User Feedback Patterns

### Long Operations
```python
def long_operation(progress=gr.Progress()):
    progress(0.1, desc="Loading model...")
    # work
    progress(0.5, desc="Processing...")
    # work
    progress(1.0, desc="Done!")
    play_completion_beep()  # Audio notification
    return result, "Success message"
```

### Error Messages
```python
return None, "Error: Clear description of what went wrong and how to fix it"
```

### Status Updates
- Never Use emoji prefixes
- Provide actionable information
- Include progress indicators

## Platform Compatibility

```python
import platform

if platform.system() == "Windows":
    # Windows-specific code
elif platform.system() == "Darwin":
    # macOS-specific code
else:
    # Linux-specific code
```

** Windows Console Encoding:**
- Avoid Unicode symbols in print() that go to console
- Use ASCII-safe alternatives: `[OK]` instead of `✓`
- UI (Gradio) can use any Unicode

## File Path Handling

```python
from pathlib import Path

# Always use Path objects
project_root = Path(__file__).parent
samples_dir = project_root / _user_config.get("samples_folder")

# Convert to string only when needed
str(file_path)
```

## Code Quality Checklist

Before submitting any code change, verify:

- [ ] Is this feature in a module? (If >50 lines)
- [ ] Does it respect user config settings?
- [ ] Does it provide progress feedback?
- [ ] Does it handle errors gracefully?
- [ ] Does it work cross-platform?
- [ ] Does it play completion notification?
- [ ] Is it documented with docstrings?

## Testing Patterns

**CRITICAL: ALL testing MUST use venv Python!**

When adding features:
1. **ALWAYS use `.\venv\Scripts\python.exe`** for all Python commands
2. **Place tests in `modules/core_components/tests/` folder** - create it if it doesn't exist
3. **Check for existing tests first** - look in `modules/tests/` before creating new ones
4. **Modify existing tests** if they're missing coverage rather than duplicating
5. **Test naming**: `test_[feature_name].py` (e.g., `test_emotion_manager.py`)
6. Test with default config
3. Test with custom folder paths
4. Test with offline mode enabled
5. Test on Windows (encoding issues!)
6. Test model loading/unloading
7. Test with notifications disabled

## Common Gotchas

**Model Path Confusion**
- Downloaded models: `models_folder` config
- Trained models: `trained_models_folder` config
- They CAN be different!

**Windows Encoding**
- Console output: ASCII only
- File I/O: UTF-8 with `encoding="utf-8"`
- UI text: Any Unicode

**Path Issues**
- Always use `Path` objects
- Use `/` in path joining (works cross-platform)
- Never hardcode `\\` or `/`

**Model Loading**
- Always unload before loading different model
- Check offline mode before downloading
- Use attention mechanism helper

## Key Functions Reference

**Audio Notification**
```python
play_completion_beep()  # Respects config setting
```

**Model Management**
```python
check_and_unload_if_different(model_id)
unload_all_models()  # Free VRAM
```

**Config**
```python
save_preference(key, value)  # Auto-saves config
```

**UI Helpers**
```python
from modules.ui_components import ui_help
format_help_html(markdown_text)
```

---

## Starting a New Feature?

1. **Plan the module structure first**
2. Create files in `modules/core_components`
3. Implement feature in module
4. Import and wire in main file
5. Test thoroughly
6. Document in this file if it's a new pattern

**Remember: We're refactoring toward modularity, not adding to the monolith!**
