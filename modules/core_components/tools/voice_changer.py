"""
Voice Changer Tab

Change the voice in an audio file to match a target voice using Chatterbox VC.
No text needed — works directly on audio.

Standalone testing:
    python -m modules.core_components.tools.voice_changer
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "modules"))

import gradio as gr
import soundfile as sf
import json
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from modules.core_components.tools.generated_output_save import (
    get_existing_wav_stems,
    parse_modal_submission,
    save_generated_output,
)
from modules.core_components.tools.output_audio_pipeline import (
    OutputAudioPipelineConfig,
    apply_generation_output_pipeline,
)
from gradio_filelister import FileLister


class VoiceChangerTool(Tool):
    """Voice Changer tool implementation."""

    config = ToolConfig(
        name="Voice Changer",
        module_name="tool_voice_changer",
        description="Change the voice in audio to match a target voice (Chatterbox VC)",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Changer tool UI."""
        components = {}

        get_sample_choices = shared_state['get_sample_choices']
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))

        with gr.TabItem("Voice Changer") as voice_changer_tab:
            components['voice_changer_tab'] = voice_changer_tab
            gr.Markdown("Change the voice in any audio to match a target voice sample. <small>(Uses Chatterbox VC — no text input needed)</small>")
            with gr.Row():
                # Left column — Target Voice Sample (matches Voice Clone layout)
                with gr.Column(scale=1):
                    gr.Markdown("### Target Voice")

                    components['target_lister'] = FileLister(
                        value=get_sample_choices(),
                        height=200,
                        show_footer=False,
                        interactive=True,
                    )

                    with gr.Row():
                        pass

                    components['target_audio'] = gr.Audio(
                        label="Target Voice Preview",
                        type="filepath",
                        interactive=False,
                        visible=True,
                        value=None,
                        elem_id="voice-convert-target-audio"
                    )

                    components['target_text'] = gr.Textbox(
                        label="Sample Text",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                    components['target_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                # Right column — Source Audio & Output (matches Voice Clone's right column)
                with gr.Column(scale=3):
                    gr.Markdown("### Source Audio")

                    components['source_audio'] = gr.Audio(
                        label="Upload or record the audio whose voice you want to change",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )

                    components['convert_btn'] = gr.Button(
                        "Convert Voice", variant="primary", size="lg"
                    )

                    components['output_audio'] = gr.Audio(
                        label="Converted Audio",
                        type="filepath",
                        interactive=True,
                    )
                    with gr.Row():
                        components['output_enable_denoise'] = gr.Checkbox(
                            label="Enable Denoise",
                            value=False,
                            visible=deepfilter_available,
                        )
                        components['output_enable_normalize'] = gr.Checkbox(
                            label="Enable Normalize",
                            value=False,
                        )
                        components['output_enable_mono'] = gr.Checkbox(
                            label="Enable Mono",
                            value=False,
                        )
                        components['output_apply_pipeline_btn'] = gr.Button(
                            "Apply Pipeline",
                            variant="secondary",
                            size="sm",
                        )

                    with gr.Row():
                        components['save_btn'] = gr.Button(
                            "Save to Output", variant="primary", interactive=False
                        )

                    components['convert_status'] = gr.Textbox(
                        label="Status", interactive=False, lines=2, max_lines=5
                    )

                    # Hidden state for metadata and temp path
                    components['temp_output_path'] = gr.State(value=None)
                    components['suggested_name'] = gr.State(value="")
                    components['metadata_text'] = gr.State(value="")
                    components['existing_files_json'] = gr.State(value="[]")

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Changer tab events."""

        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        get_tenant_output_dir = shared_state['get_tenant_output_dir']
        TEMP_DIR = shared_state['TEMP_DIR']
        play_completion_beep = shared_state.get('play_completion_beep')
        show_input_modal_js = shared_state['show_input_modal_js']
        input_trigger = shared_state['input_trigger']
        normalize_audio = shared_state['normalize_audio']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))

        tts_manager = get_tts_manager()

        def get_selected_sample_name(lister_value):
            """Extract selected sample name from FileLister value."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                from modules.core_components.tools import strip_sample_extension
                return strip_sample_extension(selected[0])
            return None

        def on_target_select(lister_value):
            """Load target voice preview, text, and info when selected."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return None, "", ""
            return load_sample_details(sample_name)

        def convert_voice(source_audio, target_lister_value, progress=gr.Progress()):
            """Run voice conversion."""
            if source_audio is None:
                return None, gr.update(interactive=False), None, "", "", "Please upload or record source audio."

            # source_audio is (sample_rate, numpy_array) from gr.Audio(type="numpy")
            src_sr, src_data = source_audio

            target_name = get_selected_sample_name(target_lister_value)
            if not target_name:
                return None, gr.update(interactive=False), None, "", "", "Please select a target voice sample."

            # Find target sample wav path
            samples = get_available_samples()

            target_wav = None
            for s in samples:
                if s["name"] == target_name:
                    target_wav = s["wav_path"]
                    break

            if not target_wav:
                return None, gr.update(interactive=False), None, "", "", f"Target sample '{target_name}' not found."

            try:
                progress(0.1, desc="Loading Chatterbox VC model...")

                # Save source numpy audio to a temp WAV (Chatterbox VC expects a file path)
                import numpy as np
                src_temp = str(TEMP_DIR / "_vc_source_input.wav")
                # Gradio may return int16/int32 — convert to float for soundfile
                if src_data.dtype in (np.int16, np.int32):
                    src_data = src_data.astype(np.float32) / np.iinfo(src_data.dtype).max
                sf.write(src_temp, src_data, src_sr)

                progress(0.4, desc="Converting voice...")
                audio_data, sr = tts_manager.generate_voice_convert_chatterbox(
                    source_audio_path=src_temp,
                    target_voice_path=target_wav,
                )

                progress(0.8, desc="Saving to temp...")

                # Save to temp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_target = "".join(c if c.isalnum() else "_" for c in target_name)
                temp_filename = f"vc_{safe_target}_{timestamp}.wav"
                temp_path = str(TEMP_DIR / temp_filename)
                sf.write(temp_path, audio_data, sr)

                suggested = f"vc_{safe_target}"

                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Type: Voice Conversion
                    Target Voice: {target_name}
                    Engine: Chatterbox VC
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()

                return (
                    temp_path,
                    gr.update(interactive=True),
                    temp_path,
                    suggested,
                    metadata_out,
                    f"Voice changed to match '{target_name}'."
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, gr.update(interactive=False), None, "", "", f"❌ Error: {str(e)}"

        components['convert_btn'].click(
            convert_voice,
            inputs=[components['source_audio'], components['target_lister']],
            outputs=[
                components['output_audio'],
                components['save_btn'],
                components['temp_output_path'],
                components['suggested_name'],
                components['metadata_text'],
                components['convert_status'],
            ]
        )

        def apply_voice_changer_output_pipeline(audio_value, enable_denoise, enable_normalize, enable_mono, request: gr.Request):
            pipeline = OutputAudioPipelineConfig(
                enable_denoise=bool(enable_denoise),
                enable_normalize=bool(enable_normalize),
                enable_mono=bool(enable_mono),
            )
            updated_audio, status = apply_generation_output_pipeline(
                audio_value,
                pipeline,
                deepfilter_available=deepfilter_available,
                denoise_step=lambda path: clean_audio(path),
                normalize_step=lambda path: normalize_audio(path, request=request),
                mono_step=lambda path: convert_to_mono(path, request=request),
            )
            if not updated_audio:
                return gr.update(), status
            return gr.update(value=updated_audio), status

        components['output_apply_pipeline_btn'].click(
            apply_voice_changer_output_pipeline,
            inputs=[
                components['output_audio'],
                components['output_enable_denoise'],
                components['output_enable_normalize'],
                components['output_enable_mono'],
            ],
            outputs=[components['output_audio'], components['convert_status']],
        )

        # Target voice preview + text + info
        components['target_lister'].change(
            on_target_select,
            inputs=[components['target_lister']],
            outputs=[components['target_audio'], components['target_text'], components['target_info']]
        )

        # Double-click = play target audio
        components['target_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#voice-convert-target-audio .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # Auto-refresh samples when tab is selected
        components['voice_changer_tab'].select(
            lambda: get_sample_choices(),
            outputs=[components['target_lister']]
        )

        # Save workflow: button → input modal → copy to output
        save_vc_modal_js = show_input_modal_js(
            title="Save Voice Changer Output",
            message="Enter a filename for this changed audio:",
            placeholder="e.g., vc_my_voice, converted_sample",
            context="save_vc_"
        )

        def get_vc_existing_files(request: gr.Request):
            output_dir = get_tenant_output_dir(request=request, strict=True)
            return json.dumps(get_existing_wav_stems(output_dir))

        save_vc_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_vc_modal_js};
            openModal(suggestedName);
        }}
        """

        components['save_btn'].click(
            fn=get_vc_existing_files,
            inputs=[],
            outputs=[components['existing_files_json']],
        ).then(
            fn=None,
            inputs=[components['existing_files_json'], components['suggested_name']],
            js=save_vc_js
        )

        def handle_vc_input_modal(input_value, edited_audio, temp_path, metadata_text, request: gr.Request):
            """Process save modal submission for Voice Changer."""
            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_vc_")
            if not matched:
                return gr.update(), gr.update()
            if cancelled:
                return gr.update(), gr.update()
            if not chosen_name or not chosen_name.strip():
                return "❌ Please enter a filename.", gr.update()

            audio_to_save = edited_audio
            if not audio_to_save and temp_path:
                audio_to_save = temp_path
            if not audio_to_save:
                return "❌ No audio to save. Please convert again.", gr.update(interactive=False)

            try:
                output_dir = get_tenant_output_dir(request=request, strict=True)
                output_path = save_generated_output(
                    audio_value=audio_to_save,
                    output_dir=output_dir,
                    raw_name=chosen_name,
                    metadata_text=metadata_text,
                    default_sample_rate=24000,
                )
                return f"Saved as {output_path.name}", gr.update(interactive=False)
            except Exception as e:
                return f"❌ Save failed: {str(e)}", gr.update()

        input_trigger.change(
            handle_vc_input_modal,
            inputs=[input_trigger, components['output_audio'], components['temp_output_path'], components['metadata_text']],
            outputs=[components['convert_status'], components['save_btn']]
        )


# Export for tab registry
get_tool_class = lambda: VoiceChangerTool


if __name__ == "__main__":
    """Standalone testing of Voice Changer tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoiceChangerTool, port=7866, title="Voice Changer - Standalone")
