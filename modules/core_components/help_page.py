"""Help Guide tab content and tool definition."""

from __future__ import annotations

from textwrap import dedent

import gradio as gr

from modules.core_components.tool_base import Tool, ToolConfig


def show_library_manager_help():
    return dedent(
        """
        ### Library Manager

        Tenant-scoped workspace for **all** sample and dataset operations.

        #### Layout
        - **Samples**: sample upload, preview, transcript autosave, cache clearing, deletion.
        - **Datasets**: dataset folder CRUD, bulk upload, per-file transcript autosave, batch transcribe, auto-split selected file.
        - **Processing Studio**: opened from Samples/Datasets only, deterministic pipeline, transcribe, context-first save.

        #### Samples flow
        1. Open **Samples**.
        2. Upload one or many audio/video files.
        3. Select a sample to preview/edit transcript (autosaves with status).
        4. Use **Process** to open it in Processing Studio.
        5. Use **Clear Cache** to remove Qwen/Lux cache files.

        #### Datasets flow
        1. Open **Datasets**.
        2. Create/select a dataset folder.
        3. Upload many files (audio or video).
        4. Edit per-file transcript (autosaves) or run **Batch Transcribe**.
        5. Use **Process** for per-file cleanup/transcribe/save workflows.
        6. Use **Auto-Split Selected File** to split directly from Datasets tab.

        #### Processing Studio flow
        1. Open source via **Process** from Samples or Datasets.
        2. Configure pipeline checkboxes (**Denoise**, **Normalize**, **Mono**) and click **Apply Pipeline**.
        3. Pipeline always recomputes from the original source file.
        4. Run **Transcribe** with Qwen3 ASR, Whisper, or VibeVoice ASR.
        5. Save with context-first primary destination (plus explicit cross-save), choosing **Save as new** or **Replace existing**.
        6. Processing transcript is written only when you save output.

        #### Quotas and tenant isolation
        - Tenant usage meter is shown at the top.
        - All reads/writes are tenant-scoped.
        - Upload and generated writes are blocked when per-file or tenant quota limits are exceeded.
        - Required tenant header is configured in Settings (default `X-Tenant-Id`) unless a default tenant fallback is configured at launch.
        """
    )


def show_voice_clone_help():
    return dedent(
        """
        ### Voice Clone

        Generate speech from your tenant sample library.

        1. Create/select samples in **Library Manager**.
        2. Choose sample in Voice Clone.
        3. Enter target text and generation settings.
        4. Generate and review output in **Output History**.
        """
    )


def show_voice_changer_help():
    return dedent(
        """
        ### Voice Changer

        Convert source audio to match a selected target sample (Chatterbox VC).

        1. Select target sample.
        2. Upload/record source audio.
        3. Convert and save output.
        """
    )


def show_voice_presets_help():
    return dedent(
        """
        ### Voice Presets

        Use Qwen preset speakers or tenant-trained voices with style control.

        1. Select voice type (preset or trained model).
        2. Enter prompt text and optional style.
        3. Generate and save output.
        """
    )


def show_conversation_help():
    return dedent(
        """
        ### Conversation

        Multi-speaker generation using Qwen/VibeVoice/other enabled engines.

        1. Prepare sample voices in **Library Manager** when using custom voice routes.
        2. Write script with speaker tags.
        3. Configure model/engine/pause settings.
        4. Generate and review in **Output History**.
        """
    )


def show_voice_design_help():
    return dedent(
        """
        ### Voice Design

        Create voices from natural-language descriptions, then reuse as samples.

        1. Enter text + voice instruction.
        2. Generate preview.
        3. Save output/sample for future cloning workflows.
        """
    )


def show_train_help():
    return dedent(
        """
        ### Train Model

        Train tenant-scoped custom voices from dataset folders.

        1. Build datasets in **Library Manager** (files + transcripts).
        2. Select dataset folder and reference clip.
        3. Configure batch/learning rate/epochs.
        4. Start training and test resulting model in **Voice Presets**.
        """
    )


def show_sound_effects_help():
    return dedent(
        """
        ### Sound Effects

        Generate audio from text prompts or video sources (MMAudio).

        1. Choose mode (Text to Audio / Video to Audio).
        2. Configure prompt, model, seed, settings.
        3. Generate and save output.
        """
    )


def show_prompt_manager_help():
    return dedent(
        """
        ### Prompt Manager

        Save prompts, generate prompts from an endpoint, and route prompts to other tabs.

        1. Manage saved prompts list.
        2. Configure endpoint/model discovery.
        3. Generate prompt text.
        4. Send prompt to destination tab fields.
        """
    )


def show_output_history_help():
    return dedent(
        """
        ### Output History

        Tenant-scoped browser for generated output files.

        1. Select output to preview and inspect metadata.
        2. Delete selected outputs when no longer needed.
        """
    )


def show_settings_help():
    return dedent(
        """
        ### Settings

        Global runtime and storage configuration.

        - This tab is visible only when app is launched with `--allow-config`.
        - Configure enabled tools, engine toggles, storage paths, tenant header/quota settings, and model downloads.
        """
    )


def show_tips_help():
    return dedent(
        """
        ### Tips & Troubleshooting

        - Prefer clean, low-noise source audio.
        - Use Library Manager Processing Studio for normalization/mono/denoise before saving.
        - If transcriptions look wrong, set language explicitly for Whisper/Qwen3 ASR.
        - If writes fail, check tenant quota meter and per-file limits.
        - For Qwen3 auto-split over long audio, review clips closely or use Whisper alignment path.
        """
    )


HELP_TOPICS = [
    "Library Manager",
    "Voice Clone",
    "Voice Changer",
    "Voice Presets",
    "Conversation",
    "Voice Design",
    "Train Model",
    "Sound Effects",
    "Prompt Manager",
    "Output History",
    "Settings",
    "Tips & Troubleshooting",
]


def get_help_text(topic: str) -> str:
    mapping = {
        "Library Manager": show_library_manager_help,
        "Voice Clone": show_voice_clone_help,
        "Voice Changer": show_voice_changer_help,
        "Voice Presets": show_voice_presets_help,
        "Conversation": show_conversation_help,
        "Voice Design": show_voice_design_help,
        "Train Model": show_train_help,
        "Sound Effects": show_sound_effects_help,
        "Prompt Manager": show_prompt_manager_help,
        "Output History": show_output_history_help,
        "Settings": show_settings_help,
        "Tips & Troubleshooting": show_tips_help,
    }
    return mapping[topic]()


# Backward-compatible names kept so older imports don't break.
def show_prep_audio_help():
    return show_library_manager_help()


def show_dataset_help():
    return show_library_manager_help()


class HelpGuideTool(Tool):
    """Help Guide top-level tab."""

    config = ToolConfig(
        name="Help Guide",
        module_name="tool_help_guide",
        description="Usage documentation for all tabs",
        enabled=True,
        category="utility",
    )

    @classmethod
    def create_tool(cls, shared_state):
        components = {}
        format_help_html = shared_state["format_help_html"]

        with gr.TabItem("Help Guide", id="tab_help_guide"):
            gr.Markdown("# Voice Clone Studio - Help Guide")
            components["help_topic"] = gr.Radio(
                choices=HELP_TOPICS,
                value="Library Manager",
                show_label=False,
                container=False,
                interactive=True,
            )
            components["help_content"] = gr.HTML(
                value=format_help_html(get_help_text("Library Manager")),
                container=True,
                padding=True,
            )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        format_help_html = shared_state["format_help_html"]

        components["help_topic"].change(
            fn=lambda topic: format_help_html(get_help_text(topic)),
            inputs=[components["help_topic"]],
            outputs=[components["help_content"]],
        )


get_tool_class = lambda: HelpGuideTool
