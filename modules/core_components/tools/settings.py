"""
Settings Tab

Configure global application settings.

Standalone testing:
    python -m modules.core_components.tools.settings
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
from pathlib import Path
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.help_page import (
    show_voice_clone_help, show_conversation_help, show_voice_presets_help,
    show_voice_design_help, show_prep_samples_help, show_finetune_help,
    show_train_help, show_tips_help
)


class SettingsTool(Tool):
    """Settings tool implementation."""

    config = ToolConfig(
        name="Settings",
        module_name="tool_settings",
        description="Application settings and preferences",
        enabled=True,
        category="utility"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Settings tool UI."""
        components = {}

        # Extract needed items from shared_state
        _user_config = shared_state.get('_user_config', {})
        format_help_html = shared_state.get('format_help_html')

        with gr.TabItem("⚙️"):
            with gr.Tabs():
                with gr.TabItem("Settings"):
                    gr.Markdown("# ⚙️ Settings")
                    gr.Markdown("Configure global application settings")

                    gr.Markdown("### Model Loading")

                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                components['settings_low_cpu_mem'] = gr.Checkbox(
                                    label="Low CPU Memory Usage (Slower loading time)",
                                    value=_user_config.get("low_cpu_mem_usage", False),
                                    info="Reduces CPU RAM usage when loading models by loading weights in smaller chunks. Tradeoff: slightly slower model loading time."
                                )

                                components['settings_attention_mechanism'] = gr.Dropdown(
                                    label="Attention Mechanism",
                                    choices=["auto", "flash_attention_2", "sdpa", "eager"],
                                    value=_user_config.get("attention_mechanism", "auto"),
                                    info="Choose attention implementation.\nAuto = fastest available. flash_attention_2 (fastest) → sdpa (fast, built-in PyTorch 2.0+) → eager (slowest, always works)"
                                )

                                with gr.Row():
                                    components['settings_audio_notifications'] = gr.Checkbox(
                                        label="Audio Notifications",
                                        value=_user_config.get("browser_notifications", True),
                                        info="Play sound when audio generation completes"
                                    )

                            with gr.Column():
                                components['settings_offline_mode'] = gr.Checkbox(
                                    label="Offline Mode (Use cached models only)",
                                    value=_user_config.get("offline_mode", False),
                                    info="When enabled, only uses models found in models folder"
                                )

                                components['model_select'] = gr.Dropdown(
                                    label="Select Model to Download",
                                    info="Download models directly to models folder (recommended for offline mode)\nWhisper cannot be auto-downloaded, copy local copy of Whisper in ./models.",
                                    choices=[
                                        "--- Qwen3-TTS Base ---",
                                        "Qwen3-TTS-12Hz-0.6B-Base",
                                        "Qwen3-TTS-12Hz-1.7B-Base",
                                        "--- Qwen3-TTS CustomVoice ---",
                                        "Qwen3-TTS-12Hz-0.6B-CustomVoice",
                                        "Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                        "--- Qwen3-TTS VoiceDesign ---",
                                        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                        "--- VibeVoice TTS ---",
                                        "VibeVoice-1.5B",
                                        "VibeVoice-Large (4-bit)",
                                        "VibeVoice-Large",
                                        "--- VibeVoice ASR ---",
                                        "VibeVoice-ASR",
                                    ],
                                    value="Qwen3-TTS-12Hz-0.6B-Base"
                                )
                                components['download_btn'] = gr.Button("Download Model", scale=1)

                                # Mapping from display names to HuggingFace model IDs
                                components['MODEL_ID_MAP'] = {
                                    "Qwen3-TTS-12Hz-0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                                    "Qwen3-TTS-12Hz-1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                    "Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                                    "Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                                    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                    "VibeVoice-1.5B": "FranckyB/VibeVoice-1.5B",
                                    "VibeVoice-Large (4-bit)": "FranckyB/VibeVoice-Large-4bit",
                                    "VibeVoice-Large": "FranckyB/VibeVoice-Large",
                                    "VibeVoice-ASR": "microsoft/VibeVoice-ASR",
                                }

                        gr.Markdown("### Folder Paths")
                        gr.Markdown("Configure where files are stored. Changes apply after clicking **Apply Changes**.")

                        # Default folder paths
                        default_folders = {
                            "samples": "samples",
                            "output": "output",
                            "datasets": "datasets",
                            "temp": "temp",
                            "models": "models"
                        }
                        components['default_folders'] = default_folders

                        # Row 1: Samples, Datasets and Output folders
                        with gr.Row():
                            with gr.Column():
                                components['settings_samples_folder'] = gr.Textbox(
                                    label="Voice Samples Folder",
                                    value=_user_config.get("samples_folder", default_folders["samples"]),
                                    info="Folder for voice sample files (.wav + .json)"
                                )
                                components['reset_samples_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_output_folder'] = gr.Textbox(
                                    label="Output Folder",
                                    value=_user_config.get("output_folder", default_folders["output"]),
                                    info="Folder for generated audio files"
                                )
                                components['reset_output_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_datasets_folder'] = gr.Textbox(
                                    label="Datasets Folder",
                                    value=_user_config.get("datasets_folder", default_folders["datasets"]),
                                    info="Folder for training/finetuning datasets"
                                )
                                components['reset_datasets_btn'] = gr.Button("Reset", size="sm")

                        # Row 2: Models and Trained Models folder
                        with gr.Row():
                            with gr.Column():
                                components['settings_models_folder'] = gr.Textbox(
                                    label="Downloaded Models Folder",
                                    value=_user_config.get("models_folder", default_folders["models"]),
                                    info="Folder for downloaded model files (Qwen, VibeVoice)"
                                )
                                components['reset_models_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                components['settings_trained_models_folder'] = gr.Textbox(
                                    label="Trained Models Folder",
                                    value=_user_config.get("trained_models_folder", default_folders["models"]),
                                    info="Folder for your custom trained models"
                                )
                                components['reset_trained_models_btn'] = gr.Button("Reset", size="sm")

                            with gr.Column():
                                gr.Markdown("")

                    with gr.Column():
                        components['apply_folders_btn'] = gr.Button("Apply Changes", variant="primary", size="lg")
                        components['settings_status'] = gr.Textbox(
                            label="Status",
                            interactive=False,
                            max_lines=10
                        )

                with gr.TabItem("Help Guide"):
                    gr.Markdown("# Voice Clone Studio - Help & Guide")

                    components['help_topic'] = gr.Radio(
                        choices=[
                            "Voice Clone",
                            "Voice Presets",
                            "Conversation",
                            "Voice Design",
                            "Prep Samples",
                            "Finetune Dataset",
                            "Train Model",
                            "Tips & Tricks"
                        ],
                        value="Voice Clone",
                        show_label=False,
                        interactive=True,
                        container=False
                    )

                    components['help_content'] = gr.HTML(
                        value=format_help_html(show_voice_clone_help()),
                        container=True,
                        padding=True
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Settings tab events."""

        # Extract needed items from shared_state
        _user_config = shared_state.get('_user_config', {})
        save_preference = shared_state.get('save_preference')
        save_config = shared_state.get('save_config')
        download_model_from_huggingface = shared_state.get('download_model_from_huggingface')
        format_help_html = shared_state.get('format_help_html')

        # Save low CPU memory setting
        components['settings_low_cpu_mem'].change(
            lambda x: save_preference("low_cpu_mem_usage", x),
            inputs=[components['settings_low_cpu_mem']],
            outputs=[]
        )
        components['settings_attention_mechanism'].change(
            lambda x: save_preference("attention_mechanism", x),
            inputs=[components['settings_attention_mechanism']],
            outputs=[]
        )

        # Save offline mode setting
        components['settings_offline_mode'].change(
            lambda x: save_preference("offline_mode", x),
            inputs=[components['settings_offline_mode']],
            outputs=[]
        )

        # Save audio notifications setting
        components['settings_audio_notifications'].change(
            lambda x: save_preference("browser_notifications", x),
            inputs=[components['settings_audio_notifications']],
            outputs=[]
        )

        # Reset button handlers
        def reset_folder(folder_key):
            return components['default_folders'][folder_key]

        components['reset_samples_btn'].click(
            lambda: reset_folder("samples"),
            outputs=[components['settings_samples_folder']]
        )

        components['reset_output_btn'].click(
            lambda: reset_folder("output"),
            outputs=[components['settings_output_folder']]
        )

        components['reset_datasets_btn'].click(
            lambda: reset_folder("datasets"),
            outputs=[components['settings_datasets_folder']]
        )

        components['reset_models_btn'].click(
            lambda: reset_folder("models"),
            outputs=[components['settings_models_folder']]
        )

        components['reset_trained_models_btn'].click(
            lambda: reset_folder("models"),
            outputs=[components['settings_trained_models_folder']]
        )

        def download_model_clicked(model_display_name):
            if not model_display_name or model_display_name.startswith("---"):
                return "❌ Please select an actual model (not a category header)"
            # Convert display name to full model ID
            model_id = components['MODEL_ID_MAP'].get(model_display_name, model_display_name)

            success, message, path = download_model_from_huggingface(model_id, progress=None)

            status = f"✓ {message}" if success else f"❌ {message}"
            return status

        # Apply folder changes
        def apply_folder_changes(samples, output, datasets, models, trained_models):
            try:
                # Get project root directory
                base_dir = Path(__file__).parent.parent.parent.parent

                # Update paths
                new_samples = base_dir / samples
                new_output = base_dir / output
                new_datasets = base_dir / datasets
                new_models = base_dir / models
                new_trained_models = base_dir / trained_models

                # Create directories if they don't exist
                new_samples.mkdir(exist_ok=True)
                new_output.mkdir(exist_ok=True)
                new_datasets.mkdir(exist_ok=True)
                new_models.mkdir(exist_ok=True)
                new_trained_models.mkdir(exist_ok=True)

                # Set HuggingFace cache environment variable
                import os
                os.environ['HF_HOME'] = str(new_models)

                # Save to config
                _user_config["samples_folder"] = samples
                _user_config["output_folder"] = output
                _user_config["datasets_folder"] = datasets
                _user_config["models_folder"] = models
                _user_config["trained_models_folder"] = trained_models
                save_config(_user_config)

                return f"Folder paths updated successfully!\n\nSamples: {new_samples}\nOutput: {new_output}\nDatasets: {new_datasets}\nDownloaded Models: {new_models}\nTrained Models: {new_trained_models}\n\nNote: Restart the app to fully apply changes to all components."

            except Exception as e:
                return f"❌ Error applying changes: {str(e)}"

        components['download_btn'].click(
            fn=download_model_clicked,
            inputs=[components['model_select']],
            outputs=[components['settings_status']]
        )

        components['apply_folders_btn'].click(
            apply_folder_changes,
            inputs=[components['settings_samples_folder'], components['settings_output_folder'], components['settings_datasets_folder'], components['settings_models_folder'], components['settings_trained_models_folder']],
            outputs=[components['settings_status']]
        )

        # Help Guide topic selector
        def show_help(topic):
            help_map = {
                "Voice Clone": show_voice_clone_help,
                "Conversation": show_conversation_help,
                "Voice Presets": show_voice_presets_help,
                "Voice Design": show_voice_design_help,
                "Prep Samples": show_prep_samples_help,
                "Finetune Dataset": show_finetune_help,
                "Train Model": show_train_help,
                "Tips & Tricks": show_tips_help
            }
            return format_help_html(help_map[topic]())

        components['help_topic'].change(
            fn=show_help,
            inputs=components['help_topic'],
            outputs=components['help_content']
        )


# Export for tab registry
get_tool_class = lambda: SettingsTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone

    # Settings needs download_model_from_huggingface function
    def mock_download_model(model_id, progress=None):
        """Mock download function for standalone testing."""
        return False, f"Download not available in standalone mode. Model: {model_id}", None

    extra_shared_state = {'download_model_from_huggingface': mock_download_model}
    run_tool_standalone(SettingsTool, port=7870, title="Settings - Standalone", extra_shared_state=extra_shared_state)
