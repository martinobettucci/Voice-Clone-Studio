"""
Train Model Tab

Train custom voice models using finetuning datasets.
"""

import gradio as gr
from textwrap import dedent
from modules.core_components.tool_base import Tool, ToolConfig


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

        with gr.TabItem("Train Model"):
            gr.Markdown("Train a custom voice model using your finetuning dataset")
            with gr.Row():
                # Left column - Dataset selection
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset Selection")

                    components['train_folder_dropdown'] = gr.Dropdown(
                        choices=["(Select Dataset)"] + get_dataset_folders(),
                        value="(Select Dataset)",
                        label="Training Dataset",
                        info="Select prepared subfolder",
                        interactive=True
                    )

                    components['refresh_train_folder_btn'] = gr.Button("Refresh Datasets", size="sm")

                    components['ref_audio_dropdown'] = gr.Dropdown(
                        choices=[],
                        label="Select Reference Audio Track",
                        info="Select one sample from your dataset as reference",
                        interactive=True
                    )

                    components['ref_audio_preview'] = gr.Audio(
                        label="Preview",
                        type="filepath",
                        interactive=False
                    )

                    components['start_training_btn'] = gr.Button("Start Training", variant="primary", size="lg")

                    train_quick_guide = dedent("""\
                        **Quick Guide:**
                        1. Select dataset folder
                        2. Enter speaker name
                        3. Choose reference audio from dataset
                        4. Configure parameters & start training (defaults work well for most cases)

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
        input_trigger = shared_state['input_trigger']
        show_input_modal_js = shared_state['show_input_modal_js']
        DATASETS_DIR = shared_state['DATASETS_DIR']

        # --- Folder change: update ref audio dropdown ---
        def update_ref_audio_dropdown(folder):
            """Update reference audio dropdown when folder changes."""
            files = get_dataset_files(folder)
            return gr.update(choices=files, value=None), None

        components['train_folder_dropdown'].change(
            update_ref_audio_dropdown,
            inputs=[components['train_folder_dropdown']],
            outputs=[components['ref_audio_dropdown'], components['ref_audio_preview']]
        )

        # --- Refresh folders ---
        components['refresh_train_folder_btn'].click(
            lambda: gr.update(choices=["(Select Dataset)"] + get_dataset_folders(), value="(Select Dataset)"),
            outputs=[components['train_folder_dropdown']]
        )

        # --- Ref audio preview ---
        def load_ref_audio_preview(folder, filename):
            """Load reference audio preview."""
            if not folder or not filename or folder in ("(No folders)", "(Select Dataset)"):
                return None
            audio_path = DATASETS_DIR / folder / filename
            if audio_path.exists():
                return str(audio_path)
            return None

        components['ref_audio_dropdown'].change(
            load_ref_audio_preview,
            inputs=[components['train_folder_dropdown'], components['ref_audio_dropdown']],
            outputs=[components['ref_audio_preview']]
        )

        # --- Start Training: 2-step modal with dynamic validation ---
        # Hidden JSON to pass existing model names to JS for validation
        components['existing_models_json'] = gr.JSON(value=[], visible=False)

        def fetch_existing_models():
            """Fetch current model list before opening modal."""
            return get_trained_model_names()

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
        def handle_train_model_input(input_value, folder, ref_audio, batch_size, lr, epochs, save_interval, progress=gr.Progress()):
            """Process input modal submission for training."""
            if not input_value or not input_value.startswith("train_model_"):
                return gr.update()

            # Format: "train_model_SpeakerName_timestamp"
            parts = input_value.split("_")
            if len(parts) >= 3:
                speaker_name = "_".join(parts[2:-1])
                return train_model(folder, speaker_name, ref_audio, batch_size, lr, epochs, save_interval, progress)

            return gr.update()

        input_trigger.change(
            handle_train_model_input,
            inputs=[input_trigger, components['train_folder_dropdown'], components['ref_audio_dropdown'],
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