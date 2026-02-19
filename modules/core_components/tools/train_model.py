"""
Train Model Tab

Train custom voice models using finetuning datasets.
"""

import gradio as gr
from textwrap import dedent
from gradio_filelister import FileLister
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.runtime import MemoryAdmissionError


class TrainModelTool(Tool):
    """Train Model tool implementation."""

    config = ToolConfig(
        name="Train Model",
        module_name="tool_train_model",
        description="Train custom voice models",
        enabled=True,
        category="training"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Train Model tool UI."""
        components = {}

        format_help_html = shared_state['format_help_html']
        get_dataset_folders = shared_state['get_dataset_folders']

        with gr.TabItem("Train Model") as train_tab:
            components['train_tab'] = train_tab
            gr.Markdown("Train a custom voice model using your finetuning dataset")
            with gr.Row():
                # Left column - Dataset selection
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Selection")

                    components['train_folder_dropdown'] = gr.Dropdown(
                        choices=["(Select Dataset)"] + get_dataset_folders(),
                        value="(Select Dataset)",
                        label="Training Dataset",
                        info="Select dataset prepared in Library Manager",
                        interactive=True
                    )

                    components['refresh_train_folder_btn'] = gr.Button("Refresh Datasets", size="sm", visible=False)

                    gr.Markdown("Reference Audio (speaker identity anchor)")
                    components['ref_audio_lister'] = FileLister(
                        value=[],
                        height=150,
                        show_footer=False,
                        interactive=True,
                    )

                    components['ref_audio_preview'] = gr.Audio(
                        label="Preview",
                        type="filepath",
                        interactive=False,
                        elem_id="train-ref-audio-preview"
                    )

                    ref_audio_help = dedent("""\
                        **Why this is required**
                        - Training still uses **all** audio + transcript pairs in your dataset.
                        - The selected reference clip is reused as a **speaker identity anchor** (`ref_audio`) for each sample.
                        - Pick a clean 3-10s clip from the same speaker for best consistency.
                    """)
                    gr.HTML(
                        value=format_help_html(ref_audio_help),
                        container=True,
                        padding=True)

                    components['start_training_btn'] = gr.Button("Start Training", variant="primary", size="lg")

                    train_quick_guide = dedent("""\
                        **Quick Guide:**
                        1. Create/prepare dataset in Library Manager
                        2. Select dataset folder
                        3. Enter speaker name
                        4. Choose one clean reference audio (identity anchor, not the only training sample)
                        5. Configure parameters & start training (defaults work well for most cases)

                        *See Help Guide tab -> Train Model for detailed instructions*
                    """)
                    gr.HTML(
                        value=format_help_html(train_quick_guide),
                        container=True,
                        padding=True)

                # Right column - Training configuration
                with gr.Column(scale=1):
                    gr.Markdown("### Training Parameters")

                    components['batch_size_slider'] = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        label="Batch Size",
                        info="Reduce if you get out of memory errors"
                    )

                    components['learning_rate_slider'] = gr.Slider(
                        minimum=1e-6,
                        maximum=1e-4,
                        value=2e-6,
                        label="Learning Rate",
                        info="Default: 2e-6"
                    )

                    components['num_epochs_slider'] = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=5,
                        step=1,
                        label="Number of Epochs",
                        info="How many times to train on the full dataset"
                    )

                    components['save_interval_slider'] = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Save Interval (Epochs)",
                        info="Save checkpoint every N epochs (0 = save every epoch)"
                    )

                    components['training_status'] = gr.Textbox(
                        label="Status",
                        lines=20,
                        interactive=False
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Train Model tab events."""

        get_dataset_files = shared_state['get_dataset_files']
        get_dataset_folders = shared_state['get_dataset_folders']
        get_trained_model_names = shared_state['get_trained_model_names']
        train_model = shared_state['train_model']
        get_tenant_datasets_dir = shared_state['get_tenant_datasets_dir']
        input_trigger = shared_state['input_trigger']
        show_input_modal_js = shared_state['show_input_modal_js']
        run_heavy_job = shared_state.get('run_heavy_job')

        def get_selected_ref_filename(lister_value):
            """Extract selected filename from FileLister value."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                return selected[0]
            return None

        # --- Folder change: update ref audio lister ---
        def update_ref_audio_lister(folder, request: gr.Request):
            """Update reference audio lister when folder changes."""
            files = get_dataset_files(folder, request=request, strict=True)
            return files, None

        components['train_folder_dropdown'].change(
            update_ref_audio_lister,
            inputs=[components['train_folder_dropdown']],
            outputs=[components['ref_audio_lister'], components['ref_audio_preview']]
        )

        # Auto-refresh datasets when tab is selected
        def refresh_train_folders(request: gr.Request):
            return gr.update(
                choices=["(Select Dataset)"] + get_dataset_folders(request=request, strict=True),
                value="(Select Dataset)",
            )

        components['train_tab'].select(
            refresh_train_folders,
            outputs=[components['train_folder_dropdown']]
        )

        # --- Ref audio preview on selection ---
        def load_ref_audio_preview(lister_value, folder, request: gr.Request):
            """Load reference audio preview from FileLister selection."""
            filename = get_selected_ref_filename(lister_value)
            if not folder or not filename or folder in ("(No folders)", "(Select Dataset)"):
                return None
            datasets_dir = get_tenant_datasets_dir(request=request, strict=True)
            audio_path = datasets_dir / folder / filename
            if audio_path.exists():
                return str(audio_path)
            return None

        components['ref_audio_lister'].change(
            load_ref_audio_preview,
            inputs=[components['ref_audio_lister'], components['train_folder_dropdown']],
            outputs=[components['ref_audio_preview']]
        )

        # Double-click = play preview
        components['ref_audio_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#train-ref-audio-preview .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # --- Start Training: 2-step modal with dynamic validation ---
        # Hidden JSON to pass existing model names to JS for validation
        components['existing_models_json'] = gr.JSON(value=[], visible=False)

        def fetch_existing_models(request: gr.Request):
            """Fetch current model list before opening modal."""
            return get_trained_model_names(request=request, strict=True)

        # Build the base modal JS using show_input_modal_js
        base_modal_js = show_input_modal_js(
            title="Start Training",
            message="Enter a name for this trained voice model:",
            placeholder="e.g., MyVoice, Female-Narrator, John-Doe",
            submit_button_text="Start Training",
            context="train_model_"
        )

        # Wrap to inject dynamic validation from existing models list
        open_modal_js = f"""
        (existingModels) => {{
            window.inputModalValidation = (value) => {{
                if (!value || value.trim().length === 0) {{
                    return 'Please enter a model name';
                }}
                if (existingModels && Array.isArray(existingModels)) {{
                    if (existingModels.includes(value.trim())) {{
                        return 'Model "' + value.trim() + '" already exists!';
                    }}
                }}
                return null;
            }};
            const openModal = {base_modal_js};
            openModal('');
        }}
        """

        components['start_training_btn'].click(
            fn=fetch_existing_models,
            inputs=None,
            outputs=[components['existing_models_json']]
        ).then(
            fn=None,
            inputs=[components['existing_models_json']],
            outputs=None,
            js=open_modal_js
        )

        # --- Handle training modal submission ---
        def handle_train_model_input(
            input_value,
            folder,
            ref_lister,
            batch_size,
            lr,
            epochs,
            save_interval,
            request: gr.Request,
            progress=gr.Progress(),
        ):
            """Process input modal submission for training."""
            if not input_value or not input_value.startswith("train_model_"):
                return gr.update()

            ref_audio = get_selected_ref_filename(ref_lister)

            # Format: "train_model_SpeakerName_timestamp"
            parts = input_value.split("_")
            if len(parts) >= 3:
                speaker_name = "_".join(parts[2:-1])
                def _run():
                    return train_model(
                        folder, speaker_name, ref_audio, batch_size, lr, epochs, save_interval,
                        progress, request=request
                    )
                try:
                    if run_heavy_job:
                        return run_heavy_job("train_model", _run, request=request)
                    return _run()
                except MemoryAdmissionError as exc:
                    return f"⚠ Memory safety guard rejected request: {str(exc)}"

            return gr.update()

        input_trigger.change(
            handle_train_model_input,
            inputs=[input_trigger, components['train_folder_dropdown'], components['ref_audio_lister'],
                    components['batch_size_slider'], components['learning_rate_slider'],
                    components['num_epochs_slider'], components['save_interval_slider']],
            outputs=[components['training_status']]
        )


# Export for tab registry
get_tool_class = lambda: TrainModelTool

if __name__ == "__main__":
    """Standalone testing of Train Model tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(TrainModelTool, port=7863, title="Train Model - Standalone")
