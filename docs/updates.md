# Version History

## February 10, 2026
#### Version 1.5.2 - Network Mode & Tweaks
- **Sentence Per Chunk**  - To prevent VibeVoice from going off the rail, we add the option of resetting after a certain amount of Sentences.
- **New System Prompt for Conversations** - Added an extra System prompt to help generate conversations.
- **Network Mode** - Now possible to launch Voice Clone Studio in network mode, so it can be access by other local machines. (In Settings)

## February 9, 2026
#### Version 1.5.0 - Prompt Manager & Emotion Storage

**Prompt Manager**
- **New Prompt Manager Tool** - Save, browse, and generate text prompts with a built-in LLM generator powered by llama.cpp
- **Saved Prompts** - Store prompts in a local `prompts.json` file with save, delete, and clear functionality
- **LLM Generation** - Generate prompts locally using Qwen3-4B or Qwen3-8B GGUF models via llama.cpp (no cloud API)
- **System Prompt Presets** - Built-in presets for TTS/Voice and Sound Design/SFX workflows, plus a custom option
- **Model Auto-Download** - Download Qwen3 models directly from HuggingFace into `models/llama/`
- **Custom Models** - Drop any `.gguf` file into `models/llama/` to use your own models
- **Automatic Server Management** - llama.cpp server starts/stops automatically, cleaned up on exit or Clear VRAM

**Standalone Emotion Storage**
- **Standalone emotions.json** - Emotion presets are now stored in a dedicated `emotions.json` file instead of inside `config.json`
- **Automatic Migration** - Existing emotions in `config.json` are automatically migrated to the new file on first launch
- **Independent Reset** - Resetting `config.json` no longer wipes saved emotion presets

**Quality of Life**
- **Clear VRAM Stops LLM** - The Clear VRAM button now also shuts down the llama.cpp server if running
- **SFX Filename Simplification** - Sound effect filenames now use the first 8 words of the prompt instead of 40-char truncation with timestamp

## February 8, 2026
#### Version 1.4.0 - Sound Effects with the addition of MMAudio

**Sound Effects (MMAudio)**
- **New Sound Effects Tool** - Generate sound effects and ambient audio using MMAudio (CVPR 2025, MIT license), supporting both text-to-audio and video-to-audio modes
- **Text-to-Audio** - Describe any sound and generate 44.1kHz audio with adjustable duration, guidance strength, and negative prompts
- **Video-to-Audio** - Drop in a video clip and MMAudio generates matching sound effects synchronized to the visual content
- **Multiple Model Sizes** - Choose between Medium (2.4GB) and Large v2 (3.9GB) built-in models, with support for custom models
- **Custom Model Support** - Load your own `.pth` or `.safetensors` MMAudio checkpoints with automatic architecture detection
- **Video Preview** - Source/Result toggle to compare original video against the generated audio-muxed result

## February 8, 2026
#### Version 1.3.0 - Auto-Split Audio, Dataset Management & Engine Controls

**Auto-Split Audio**
- **Automatic Audio Splitting** - Split long audio files into clean sentence-level clips using Qwen3 or Whisper's timestamp extraction.
- **One-Click Dataset Creation** - Split audio and auto-save segments with transcripts directly into dataset folders
- **Trim and discard Silent areas** - Uses the timestamp data to find and remove non verbal moments.

**Unified ASR Engine**
- **Unified ASR Dropdown** - Single dropdown for all transcription engines (Qwen3 ASR, VibeVoice ASR, Whisper) replacing the old radio + size selector
- **ASR Engine Toggles** - Enable or disable individual ASR engines in Settings, just like TTS engines
- **Dynamic Defaults** - ASR dropdown automatically picks the best available engine based on what's installed and enabled
- **Added Whisper Large** - With the addition of automatic disply of available ASR engine, adding more choices doesn't bloat the ui.

**Engine Availability Checker**
- **Startup Engine Detection** - App now auto-checks which TTS and ASR engines are installed at launch
- **Auto-Disable Missing Engines** - Engines that aren't installed are automatically hidden from dropdowns
- **Clean Console Output** - Clear status report showing which engines are available, skipped, or missing

**Dataset Management**
- **Create Dataset Folders** - Create new dataset folders directly from the Prep Audio UI
- **Manage Existing Datasets** - Delete dataset folders with confirmation modal
- **Drag & Drop Audio** - Import audio files by dragging them into the editor

**Quality of Life**
- **Renamed "Qwen CustomVoice" to "Qwen Speakers"** - Clearer label in Conversation and Voice Presets tabs
- **Overwrite Protection** - Inline confirmation bar when saving a file that already exists
- **Friendly Port Error** - Clean message when Voice Clone Studio is already running instead of a traceback
- **Whisper Now Optional** - Moved from auto-install to optional in setup wizard, same as Qwen3-ASR.
- **Suppressed Noisy Warnings** - Silenced verbose k2 and flash-attn warnings during engine checks

## February 8, 2026
#### Version 1.2.0 - Qwen3-ASR & ICL Support for Trained Models
- **Qwen3 ASR Integration** - Added Qwen3-ASR as a new transcription engine in Prep Audio, supporting 52 languages and dialects
- **Model Size Selector** - Choose between Small (0.6B, fast) and Large (1.7B, best accuracy) Qwen3 ASR models
- **Language Selection** - Qwen3 ASR supports language hints for improved accuracy, shared with Whisper's language dropdown
- **ICL (In-Context Learning) for Trained Models** - Enhanced Voice Presets with optional ICL mode that provides real-time prosody and style cues on top of trained voice identity
- **Dataset-Based ICL Samples** - Select reference audio from your training datasets for ICL, with audio preview and automatic transcript loading
- **Speaker Encoder Transplant** - Automatic fix for trained model checkpoints missing speaker encoder weights, loading them from the matching base model at runtime
- **Setup Script Integration** - Qwen3 ASR offered as optional install in setup-windows.bat, setup-linux.sh, and Dockerfile
- **Suppressed Gradio HTTP Logs** - Silenced noisy httpx/httpcore info-level logs from Gradio 6

## February 7, 2026
#### Version 1.1.0 - Added support for LuxTTS
- Added to Voice Cloning tab.
- Added to Conversation Tab. (By stiching together multiple generations)
- Forces LuxTTS to use our pre-trancribed text files. Bypassing its internal transcribe step.
- Creates caches for each sample used. Making it faster on the next run.

## February 7, 2026

#### Version 1.0.0 - Complete Modular Rewrite
- **Full Modular Architecture** - Complete rewrite from a 6000+ line monolith into independent tool modules under `modules/core_components/tools/`
- **Tool System** - Each tab is now a self-contained tool with its own UI, events, and logic, loaded dynamically from a central registry
- **Enable/Disable Tools** - New "Visible Tools" section in Settings lets you toggle any tab on or off (persisted in config, takes effect on restart)
- **Simplified Prep Audio** - Formerly "Prep Samples", now serves dual purpose for both voice sample preparation and dataset creation in a single unified tool
- **Improved FileLister Component** - Custom Gradio component (v0.4.0) with multi-select for batch file deletion and double-click to instantly play audio
- **Help Guide in Settings** - Help documentation moved into the Settings tab as a sub-tab, keeping the main tab bar clean
- **Settings Tab Right-Aligned** - Settings gear icon pushed to the far right of the tab bar for quick access
- **Centralized Constants** - All model sizes, languages, speakers, and defaults defined once in `constants.py`
- **AI Model Managers** - Centralized TTS and ASR model management with automatic VRAM optimization and model switching
- **Shared State Architecture** - Tools receive configuration, utilities, and managers through a unified shared state, enabling independent testing
- **Cleaned Up Project** - Removed obsolete documentation, stale files, and migration artifacts

## January 30, 2026

#### Version 0.7.6 - Advanced Parameters & Emotion Presets
- Bug Fixes

## January 30, 2026

#### Version 0.7.5 - Advanced Parameters & Emotion Presets
- **Added option to save Emotions** - Improved Emotion system by allowing the user to create and save their own preset.
- **Moved the Emotion Manager to Modules** - Plans are to split the app into modules, the emotion manager is the first one.

#### Version 0.7.0 - Advanced Parameters & Emotion Presets
- **Advanced Parameter Controls** - Full access to sampling parameters (temperature, top_k, top_p, repetition_penalty, max_new_tokens) across all tabs
- **Voice Clone Tab** - Emotion presets with intensity slider for Qwen models
- **Voice Presets Tab** - Context-aware controls: style instructions for Premium Speakers, emotion presets for Trained Models
- **Conversation Tab** - Model-specific advanced parameters: VibeVoice diffusion controls (CFG, inference steps, LM sampling), Qwen sampling parameters
- **Voice Design Tab** - Advanced Qwen parameters for fine-tuning voice generation
- **Emotion Preset System** - Added emotion presets with intensity control - Inspired by [Qwen3-TTS Emotional Voice Clone](https://github.com/Dawizzer/ComfyUI-Qwen3TTS-Emotional)
- **Offline Mode** - Added offline mode toggle in Settings to use locally cached models without internet access
- **Model Download Tool** - Direct model download to `models/` folder with progress tracking in console
- **Training Script Fix** - Added attention mechanism fallback (flash_attention_2 → sdpa → eager) to prevent failures without flash-attn

## January 27, 2026

#### Version 0.6.5 - Improved Conversation Tool
- **Three Conversation Modes** - Qwen CustomVoice (9 preset speakers), Qwen Base (8 custom voice samples), and VibeVoice (4 custom voice samples)
- **VRAM Optimization** - Automatic model unloading when switching between conversation modes
- **Global Model Management** - Unload All Models button for manual VRAM cleanup
- **Voice Sample Persistence** - Dropdowns remember your selected voice samples across sessions
- **Improved Conversation Options** - Inspired by [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)

#### Version 0.6.0 - Enhanced Model Support & Settings
- **VibeVoice Large 4-bit** - Added support for quantized 4-bit VibeVoice Large model for reduced VRAM usage
- **Settings Tab** - New centralized settings interface with configurable folder paths
- **Attention Mechanism Selection** - Choose between SAGE (fastest), Flash Attention 2, SDPA, or Eager with automatic fallback
- **Low CPU Memory Option** - Toggle to reduce CPU memory usage during model loading (all models)
- **UI Improvements** - Reorganized Voice Clone tab with conditional visibility and better layout
- **Refresh Voice Samples** - Added button in Conversation tab to refresh voice sample dropdowns

#### Version 0.5.5 - Platform Support & Requirements
- Consolidated requirements into single universal `requirements.txt` with platform markers
- Added platform-specific setup scripts (Windows/Linux)
- Enhanced setup automation

## January 26, 2026

#### Version 0.5.1 - Training Infrastructure
- Removed bundled Qwen3-TTS module (now using PyPI package)
- Added Qwen fine-tuning scripts for custom voice training
- Checkpoint management with configurable save intervals

#### Version 0.5.0 - UI Polish & Help System
- **Help Guide Tab** - Comprehensive in-app documentation with 8 topic sections
- **Modular Help System** - Extracted help content to separate `ui_help.py` module
- **Better Text Formatting** - Markdown rendering with scrollable containers
- gr.HTML styling improvements with container/padding support
- Label color matching enhancements

#### Version 0.4.0 - Custom Voice Training
- Added **Train Model** tab for fine-tuning custom voices
- Complete training pipeline with validation, data preparation, and model training
- **Batch Transcription** - Process 50-100+ audio files in one click
- Support for both 0.6B and 1.7B base models
- Real-time training progress monitoring with live loss values
- Checkpoint management - compare different training epochs
- Integration with Voice Presets tab for using trained models
- Dataset organization system with `datasets/` folder structure
- Automatic audio format conversion (24kHz 16-bit mono)

## January 25, 2026

#### Version 0.3.5 - Style Instructions
- Added Style Instructions support in Conversation for Qwen model (Unsupported by VibeVoice)

#### Version 0.3.0 - Enhanced Media Support
- **Video File Support** - Upload video files (.mp4, .mov, .avi, .mkv, etc.) to Prep Samples tab
- **Automatic Audio Extraction** - Uses ffmpeg to extract audio from video files for voice cloning
- **Improved Workflow** - Added Clear button to quickly reset the audio editor
- Enhanced media handling and file upload capabilities

## January 24, 2026

#### Version 0.2.0 - VibeVoice Integration
- Added **VibeVoice TTS** support for long-form multi-speaker generation (up to 90 minutes)
- Added **VibeVoice ASR** as alternative transcription engine alongside Whisper
- Conversation tab now supports both Qwen (9 preset voices) and VibeVoice (custom samples) engines
- Multi-speaker conversation support with up to 4 custom voices
- Added Output History management
- Removed Clone Design tab

## January 23, 2026

#### Version 0.1.0 - Initial Release
- Voice cloning with Qwen3-TTS (Base, CustomVoice, VoiceDesign models)
- Whisper-powered automatic transcription
- Sample preparation toolkit (trim, normalize, mono conversion)
- Voice prompt caching for faster generation
- Seed control for reproducible outputs
- 9 premium preset voices (Qwen CustomVoice)
- Voice Design from text descriptions
- Conversation mode with multi-speaker support
