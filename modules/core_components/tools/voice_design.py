"""
Voice Design Tab

Create new voices from natural language descriptions using Qwen3-TTS VoiceDesign.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
import gradio as gr
import soundfile as sf
import shutil
import json
import torch
import random
from datetime import datetime
from pathlib import Path
from modules.core_components.ai_models.model_utils import set_seed

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from modules.core_components.ui_components.prompt_assistant import (
    create_prompt_assistant,
    wire_prompt_assistant_events,
    wire_prompt_apply_listener,
)


class VoiceDesignTool(Tool):
    """Voice Design tool implementation."""

    config = ToolConfig(
        name="Voice Design",
        module_name="tool_voice_design",
        description="Create voices from natural language descriptions",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Design tool UI."""
        components = {}

        # Get shared utilities
        LANGUAGES = shared_state.get('LANGUAGES', ['Auto'])
        _user_config = shared_state.get('_user_config', {})
        create_qwen_advanced_params = shared_state.get('create_qwen_advanced_params')

        with gr.TabItem("Voice Design", id="tab_voice_design"):
            gr.Markdown("Create new voices from natural language descriptions")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Create Design")

                    components['design_text_input'] = gr.Textbox(
                        label="Reference Text",
                        placeholder="Enter the text for the voice design (this will be spoken in the designed voice)...",
                        lines=3,
                        value="Thank you for listening to this voice design sample. This sentence is intentionally a bit long so you can hear the full range and quality of the generated voice."
                    )

                    components['design_instruct_input'] = gr.Textbox(
                        label="Voice Design Instructions",
                        placeholder="Describe the voice: e.g., 'Young female voice, bright and cheerful, slightly breathy' or 'Deep male voice with a warm, comforting tone, speak slowly'",
                        lines=3
                    )
                    components['prompt_assistant'] = create_prompt_assistant(
                        shared_state=shared_state,
                        target_ids=["voice_design.reference", "voice_design.instructions"],
                        default_target_id="voice_design.reference",
                    )

                    with gr.Row():
                        components['design_language'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                            scale=2
                        )
                        components['design_seed'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    components['save_to_output_checkbox'] = gr.Checkbox(
                        label="Save to Output folder instead of Temp",
                        value=False
                    )

                    # Qwen Advanced Parameters
                    design_params = create_qwen_advanced_params(
                        include_emotion=False,
                        visible=True,
                        shared_state=shared_state
                    )
                    components.update(design_params)

                    components['design_generate_btn'] = gr.Button("Generate Voice", variant="primary", size="lg")
                    components['design_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=3)

                with gr.Column(scale=1):
                    gr.Markdown("### Preview & Save")
                    components['design_output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        interactive=False,
                    )

                    components['design_save_btn'] = gr.Button("Save Sample", variant="primary", interactive=False)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Design events."""

        # Get shared utilities and directories
        show_input_modal_js = shared_state.get('show_input_modal_js')
        save_preference = shared_state.get('save_preference')
        input_trigger = shared_state.get('input_trigger')
        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        get_tenant_samples_dir = shared_state.get('get_tenant_samples_dir')
        get_tenant_output_dir = shared_state.get('get_tenant_output_dir')
        get_tenant_paths = shared_state.get('get_tenant_paths')
        configure_tts_manager_for_tenant = shared_state.get('configure_tts_manager_for_tenant')
        play_completion_beep = shared_state.get('play_completion_beep')

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        def generate_voice_design_handler(text_to_generate, language, instruct, seed, save_to_output,
                                          do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                                          request: gr.Request = None,
                                          progress=gr.Progress()):
            """Generate audio using voice design with natural language instructions."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate.", gr.update()

            if not instruct or not instruct.strip():
                return None, "❌ Please enter voice design instructions.", gr.update()

            try:
                # Set the seed for reproducibility
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)

                set_seed(seed)
                seed_msg = f"Seed: {seed}"
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.1, desc="Loading VoiceDesign model...")

                # Call manager to generate
                audio_data, sr = tts_manager.generate_voice_design(
                    text=text_to_generate.strip(),
                    language=language if language != "Auto" else "Auto",
                    instruct=instruct.strip(),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens
                )

                progress(0.8, desc=f"Saving audio ({'output' if save_to_output else 'temp'})...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Ensure audio is numpy array (move to CPU if tensor)
                if hasattr(audio_data, "cpu"):
                    audio_data = audio_data.cpu().numpy()
                elif hasattr(audio_data, "numpy"):
                    audio_data = audio_data.numpy()

                if save_to_output:
                    OUTPUT_DIR = get_tenant_output_dir(request=request, strict=True)
                    out_file = OUTPUT_DIR / f"voice_design_{timestamp}.wav"
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                else:
                    # Temporary previews must stay out of output history.
                    tenant_paths = get_tenant_paths(request=request, strict=True)
                    TEMP_DIR = tenant_paths.temp_dir
                    out_file = TEMP_DIR / f"temp_voice_design_{timestamp}.wav"
                    TEMP_DIR.mkdir(parents=True, exist_ok=True)

                sf.write(str(out_file), audio_data, sr)

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return str(out_file), f"Voice design generated. Save to samples to keep.\n{seed_msg}", gr.update(interactive=True)

            except Exception as e:
                return None, f"❌ Error generating audio: {str(e)}", gr.update(interactive=False)

        def save_designed_voice(audio, design_name, ref_text, instruct, seed, request: gr.Request = None):
            """Save a designed voice as a sample (tool-specific implementation)."""
            if not audio:
                return "❌ No audio to save"

            if not design_name or not design_name.strip():
                return "❌ Please enter a name"

            try:
                # Clean the name
                safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in design_name.strip())

                # Copy audio file to samples folder
                SAMPLES_DIR = get_tenant_samples_dir(request=request, strict=True)
                SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                wav_path = SAMPLES_DIR / f"{safe_name}.wav"
                json_path = SAMPLES_DIR / f"{safe_name}.json"

                # If audio is a path, copy it
                if isinstance(audio, str):
                    shutil.copy(audio, wav_path)
                else:
                    # If audio is data, save it
                    sf.write(str(wav_path), audio, 24000)

                # Create metadata
                metadata = {
                    "Name": safe_name,
                    "Text": ref_text if ref_text else "Voice design sample",
                    "Type": "Voice Design",
                    "Instruct": instruct if instruct else "",
                    "Seed": seed if seed is not None else -1,
                    "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                return f"✓ Saved as sample: {safe_name}"

            except Exception as e:
                return f"❌ Error saving: {str(e)}"

        # Wire generate button
        components['design_generate_btn'].click(
            generate_voice_design_handler,
            inputs=[components['design_text_input'], components['design_language'], components['design_instruct_input'],
                    components['design_seed'], components['save_to_output_checkbox'],
                    components['do_sample'], components['temperature'], components['top_k'],
                    components['top_p'], components['repetition_penalty'], components['max_new_tokens']],
            outputs=[components['design_output_audio'], components['design_status'], components['design_save_btn']]
        )

        if components.get('prompt_assistant'):
            wire_prompt_assistant_events(
                assistant=components['prompt_assistant'],
                target_components={
                    "voice_design.reference": components['design_text_input'],
                    "voice_design.instructions": components['design_instruct_input'],
                },
                status_component=components['design_status'],
                shared_state=shared_state,
            )

        if prompt_apply_trigger is not None:
            wire_prompt_apply_listener(
                prompt_apply_trigger=prompt_apply_trigger,
                target_components={
                    "voice_design.reference": components['design_text_input'],
                    "voice_design.instructions": components['design_instruct_input'],
                },
                status_component=components['design_status'],
            )

        # Save designed voice - show modal
        components['design_save_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_input_modal_js(
                title="Save Designed Voice",
                message="Enter a name for this voice design:",
                placeholder="e.g., Bright-Female, Deep-Male, Cheerful-Voice",
                context="save_design_"
            )
        )

        # Handle save designed voice input modal submission
        if input_trigger:
            def handle_save_design_input(input_value, audio, ref_text, instruct, seed, request: gr.Request = None):
                """Process input modal submission for saving designed voice."""
                # Context filtering: only process if this is our context
                if not input_value or not input_value.startswith("save_design_"):
                    return gr.update()

                # Extract design name from context prefix
                # Format: "save_design_<name>_<timestamp>"
                parts = input_value.split("_")
                if len(parts) >= 3:
                    # Remove context prefix and timestamp (last part)
                    design_name = "_".join(parts[2:-1])
                    status = save_designed_voice(audio, design_name, ref_text, instruct, seed, request=request)
                    return status, gr.update(interactive=False)

                return gr.update(), gr.update()

            input_trigger.change(
                handle_save_design_input,
                inputs=[input_trigger, components['design_output_audio'], 
                        components['design_text_input'], components['design_instruct_input'],
                        components['design_seed']],
                outputs=[components['design_status'], components['design_save_btn']]
            )

        # Save language preference
        if save_preference:
            components['design_language'].change(
                lambda x: save_preference("language", x),
                inputs=[components['design_language']],
                outputs=[]
            )


# Export for registry
get_tool_class = lambda: VoiceDesignTool


if __name__ == "__main__":
    """Standalone testing of Voice Design tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoiceDesignTool, port=7861, title="Voice Design - Standalone")
