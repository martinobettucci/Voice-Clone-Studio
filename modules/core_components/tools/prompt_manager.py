"""Prompt Manager tool with endpoint-based generation and cross-tab prompt routing."""

import json
import re

import requests

import gradio as gr

import modules.core_components.prompt_hub as prompt_hub
from modules.core_components.tool_base import Tool, ToolConfig


class PromptManagerTool(Tool):
    """Prompt Manager - save, browse, generate, and route prompts."""

    config = ToolConfig(
        name="Prompt Manager",
        module_name="prompt_manager",
        description="Save, browse, and generate text prompts using an LLM",
        enabled=True,
        category="utility",
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create the Prompt Manager UI."""
        from gradio_filelister import FileLister

        user_config = shared_state.get("_user_config", {})
        use_local_ollama = bool(user_config.get("llm_use_local_ollama", False))
        saved_endpoint = prompt_hub.normalize_v1_base_url(
            str(user_config.get("llm_endpoint_url", prompt_hub.DEFAULT_OPENAI_ENDPOINT)),
            fallback=prompt_hub.DEFAULT_OPENAI_ENDPOINT,
        )
        ollama_endpoint = prompt_hub.normalize_v1_base_url(
            str(user_config.get("llm_ollama_url", prompt_hub.DEFAULT_OLLAMA_ENDPOINT)),
            fallback=prompt_hub.DEFAULT_OLLAMA_ENDPOINT,
        )
        displayed_endpoint = ollama_endpoint if use_local_ollama else saved_endpoint

        saved_llm_model = str(user_config.get("llm_model", "")).strip() or "gpt-4o-mini"

        target_choices = prompt_hub.get_enabled_target_choices(user_config)
        default_target = target_choices[0][1] if target_choices else None

        cross_tab_available = shared_state.get("prompt_apply_trigger") is not None

        components = {}

        with gr.TabItem("Prompt Manager", id="tab_prompt_manager"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Prompt Result")

                    components["prompt_text"] = gr.Textbox(
                        label="Prompt",
                        lines=8,
                        placeholder="Write your prompt here, or select a saved one from the list...",
                        interactive=True,
                    )

                    with gr.Row():
                        components["save_btn"] = gr.Button("Save Prompt", variant="primary", scale=1)
                        components["delete_btn"] = gr.Button("Delete", variant="stop", scale=1)
                        components["clear_btn"] = gr.Button("Clear", scale=1)

                    components["pm_status"] = gr.Textbox(label="Status", interactive=False, lines=1)

                    gr.Markdown("### Send To Tab")
                    components["pm_target"] = gr.Dropdown(
                        label="Target Field",
                        choices=target_choices,
                        value=default_target,
                        interactive=cross_tab_available and bool(target_choices),
                    )

                    with gr.Row():
                        components["send_replace_btn"] = gr.Button(
                            "Send (Replace)",
                            variant="primary",
                            interactive=cross_tab_available and bool(target_choices),
                        )
                        components["send_append_btn"] = gr.Button(
                            "Send (Append)",
                            interactive=cross_tab_available and bool(target_choices),
                        )

                    gr.Markdown("### Saved Prompts")
                    components["prompt_lister"] = FileLister(
                        value=prompt_hub.get_prompt_names(),
                        height=250,
                        show_footer=False,
                        interactive=True,
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### LLM Prompt Generator")

                    components["llm_instruction"] = gr.Textbox(
                        label="Instructions for LLM",
                        lines=4,
                        placeholder=(
                            "Describe what kind of prompt you want the LLM to generate...\n"
                            "e.g., 'A dramatic monologue about a pirate finding treasure'"
                        ),
                        interactive=True,
                    )

                    components["system_prompt_preset"] = gr.Dropdown(
                        label="System Prompt Preset",
                        choices=prompt_hub.SYSTEM_PROMPT_CHOICES,
                        value=prompt_hub.SYSTEM_PROMPT_CHOICES[0],
                        interactive=True,
                    )

                    components["system_prompt"] = gr.Textbox(
                        label="System Prompt",
                        lines=4,
                        value=prompt_hub.SYSTEM_PROMPTS[prompt_hub.SYSTEM_PROMPT_CHOICES[0]],
                        interactive=True,
                    )

                    components["llm_use_local_ollama"] = gr.Checkbox(
                        label="Use local Ollama",
                        value=use_local_ollama,
                        info="When enabled, Prompt Manager uses local Ollama and ignores API key auth.",
                        interactive=True,
                    )

                    components["llm_endpoint_url"] = gr.Textbox(
                        label="Endpoint URL",
                        value=displayed_endpoint,
                        info="OpenAI-compatible base URL (auto-set to Ollama URL when enabled).",
                        interactive=not use_local_ollama,
                    )

                    with gr.Row():
                        components["llm_model"] = gr.Textbox(
                            label="LLM Model",
                            value=saved_llm_model,
                            placeholder="e.g. gpt-4o-mini or llama3.1",
                            interactive=True,
                            scale=2,
                        )
                        components["refresh_models_btn"] = gr.Button("Refresh Models", scale=1)

                    components["llm_model_suggestions"] = gr.Dropdown(
                        label="Model Suggestions",
                        choices=[],
                        value=None,
                        allow_custom_value=False,
                        interactive=True,
                    )

                    components["generate_btn"] = gr.Button("Generate Prompt", variant="primary")

                    components["llm_status"] = gr.Textbox(label="LLM Status", interactive=False, lines=2)

            components["pm_existing_files_json"] = gr.Textbox(visible=False)
            components["pm_suggested_name"] = gr.Textbox(visible=False)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Prompt Manager events."""
        user_config = shared_state.get("_user_config", {})
        save_preference = shared_state.get("save_preference")
        show_input_modal_js = shared_state.get("show_input_modal_js")
        show_confirmation_modal_js = shared_state.get("show_confirmation_modal_js")
        input_trigger = shared_state.get("input_trigger")
        confirm_trigger = shared_state.get("confirm_trigger")
        prompt_apply_trigger = shared_state.get("prompt_apply_trigger")
        main_tabs_component = shared_state.get("main_tabs_component")

        prompt_get_names = shared_state.get("prompt_get_names", prompt_hub.get_prompt_names)
        prompt_build_apply_payload = shared_state.get("prompt_build_apply_payload", prompt_hub.build_apply_payload)
        prompt_get_target_tab_id = shared_state.get("prompt_get_target_tab_id", prompt_hub.get_target_tab_id)
        prompt_get_enabled_target_choices = shared_state.get("prompt_get_enabled_target_choices", prompt_hub.get_enabled_target_choices)

        def _load_prompts():
            return prompt_hub.load_prompts()

        def _save_prompts(prompts):
            return prompt_hub.save_prompts(prompts)

        def load_selected_prompt(lister_value):
            if not lister_value:
                return gr.update()
            selected = lister_value.get("selected", [])
            if len(selected) != 1:
                return gr.update()
            return gr.update(value=_load_prompts().get(selected[0], ""))

        components["prompt_lister"].change(
            load_selected_prompt,
            inputs=[components["prompt_lister"]],
            outputs=[components["prompt_text"]],
        )

        components["clear_btn"].click(
            fn=lambda: (gr.update(value=""), ""),
            outputs=[components["prompt_text"], components["pm_status"]],
        )

        save_modal_js = show_input_modal_js(
            title="Save Prompt",
            message="Enter a name for this prompt:",
            placeholder="e.g., Dramatic Pirate, Thunder Storm, Calm Narrator",
            context="save_prompt_",
        )

        def get_existing_prompt_names():
            return json.dumps(prompt_get_names())

        def get_selected_name(lister_value):
            if lister_value:
                selected = lister_value.get("selected", [])
                if len(selected) == 1:
                    return selected[0]
            return ""

        save_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_modal_js};
            openModal(suggestedName);
        }}
        """

        components["save_btn"].click(
            fn=lambda lister_val: (get_existing_prompt_names(), get_selected_name(lister_val)),
            inputs=[components["prompt_lister"]],
            outputs=[components["pm_existing_files_json"], components["pm_suggested_name"]],
        ).then(
            fn=None,
            inputs=[components["pm_existing_files_json"], components["pm_suggested_name"]],
            js=save_js,
        )

        def handle_save_prompt(input_value, prompt_text):
            no_update = gr.update(), gr.update()
            if not input_value or not input_value.startswith("save_prompt_"):
                return no_update

            parts = input_value.split("_", 3)
            if len(parts) >= 3 and parts[2] == "cancel":
                return no_update

            raw_name = input_value[len("save_prompt_") :]
            name_parts = raw_name.rsplit("_", 1)
            chosen_name = name_parts[0] if len(name_parts) > 1 else raw_name

            if not chosen_name.strip():
                return "No name provided", gr.update()
            if not prompt_text or not prompt_text.strip():
                return "No prompt text to save", gr.update()

            try:
                prompts = _load_prompts()
                prompts[chosen_name.strip()] = prompt_text.strip()
                _save_prompts(prompts)
                return f"Saved: {chosen_name.strip()}", gr.update(value=prompt_get_names())
            except Exception as e:
                return f"Error saving: {e}", gr.update()

        input_trigger.change(
            handle_save_prompt,
            inputs=[input_trigger, components["prompt_text"]],
            outputs=[components["pm_status"], components["prompt_lister"]],
        )

        components["delete_btn"].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Prompt?",
                message="This will permanently delete the selected prompt.",
                confirm_button_text="Delete",
                context="delete_prompt_",
            ),
        )

        def handle_delete_prompt(confirm_value, lister_value):
            no_update = gr.update(), gr.update(), gr.update()
            if not confirm_value or not confirm_value.startswith("delete_prompt_"):
                return no_update
            if "cancel" in confirm_value:
                return no_update

            if not lister_value:
                return "No prompt selected", gr.update(), gr.update()
            selected = lister_value.get("selected", [])
            if len(selected) != 1:
                return "Select a single prompt to delete", gr.update(), gr.update()

            prompt_name = selected[0]
            try:
                prompts = _load_prompts()
                if prompt_name not in prompts:
                    return f"Prompt not found: {prompt_name}", gr.update(), gr.update()
                del prompts[prompt_name]
                _save_prompts(prompts)
                return f"Deleted: {prompt_name}", gr.update(value=prompt_get_names()), gr.update(value="")
            except Exception as e:
                return f"Error deleting: {e}", gr.update(), gr.update()

        confirm_trigger.change(
            handle_delete_prompt,
            inputs=[confirm_trigger, components["prompt_lister"]],
            outputs=[components["pm_status"], components["prompt_lister"], components["prompt_text"]],
        )

        def on_preset_change(preset_name):
            if preset_name in prompt_hub.SYSTEM_PROMPTS:
                return gr.update(value=prompt_hub.SYSTEM_PROMPTS[preset_name])
            return gr.update()

        components["system_prompt_preset"].change(
            on_preset_change,
            inputs=[components["system_prompt_preset"]],
            outputs=[components["system_prompt"]],
        )

        def refresh_models(use_local_ollama, endpoint_url, current_model):
            models, status = prompt_hub.discover_available_models(user_config, bool(use_local_ollama), endpoint_url)
            if not models:
                return gr.update(choices=[], value=None), status, gr.update()

            selected = current_model.strip() if current_model and current_model.strip() else models[0]
            selected_dropdown = selected if selected in models else models[0]
            model_update = gr.update()
            if not current_model or not current_model.strip():
                model_update = gr.update(value=models[0])
                save_preference("llm_model", models[0])

            return gr.update(choices=models, value=selected_dropdown), status, model_update

        def on_use_local_ollama_change(use_local_ollama, endpoint_url, current_model):
            use_local = bool(use_local_ollama)
            save_preference("llm_use_local_ollama", use_local)

            if use_local:
                endpoint_value = prompt_hub.normalize_v1_base_url(
                    str(user_config.get("llm_ollama_url", prompt_hub.DEFAULT_OLLAMA_ENDPOINT)),
                    fallback=prompt_hub.DEFAULT_OLLAMA_ENDPOINT,
                )
            else:
                endpoint_value = prompt_hub.normalize_v1_base_url(
                    str(user_config.get("llm_endpoint_url", endpoint_url or prompt_hub.DEFAULT_OPENAI_ENDPOINT)),
                    fallback=prompt_hub.DEFAULT_OPENAI_ENDPOINT,
                )

            suggestions_update, status, model_update = refresh_models(use_local, endpoint_value, current_model)
            endpoint_update = gr.update(value=endpoint_value, interactive=not use_local)
            return endpoint_update, suggestions_update, status, model_update

        def on_endpoint_change(endpoint_url):
            normalized = prompt_hub.normalize_v1_base_url(endpoint_url, fallback=prompt_hub.DEFAULT_OPENAI_ENDPOINT)
            save_preference("llm_endpoint_url", normalized)
            return gr.update(value=normalized)

        def on_model_change(model_name):
            model = (model_name or "").strip()
            if model:
                save_preference("llm_model", model)

        def on_model_suggestion_change(suggested_model):
            if not suggested_model:
                return gr.update(), gr.update()
            save_preference("llm_model", suggested_model)
            return gr.update(value=suggested_model), f"Selected model: {suggested_model}"

        def generate_with_llm(instruction, system_prompt, model_name, use_local_ollama, endpoint_url, progress=gr.Progress()):
            if not instruction or not instruction.strip():
                return gr.update(), "Please enter instructions for the LLM"

            model = (model_name or "").strip()
            if not model:
                return gr.update(), "Please enter a model name"

            use_local = bool(use_local_ollama)
            base_url = prompt_hub.get_effective_base_url(use_local, endpoint_url, user_config)
            if not use_local:
                save_preference("llm_endpoint_url", base_url)
            save_preference("llm_model", model)

            try:
                progress(0.2, desc="Sending request...")
                effective_system = (
                    system_prompt.strip()
                    if system_prompt and system_prompt.strip()
                    else prompt_hub.SYSTEM_PROMPTS[prompt_hub.SYSTEM_PROMPT_CHOICES[0]]
                )
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": effective_system},
                        {"role": "user", "content": instruction.strip()},
                    ],
                    "stream": False,
                    "temperature": 0.8,
                    "top_p": 0.95,
                }

                response = requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=prompt_hub.build_headers(user_config, use_local),
                    timeout=120,
                )
                if response.status_code != 200:
                    return gr.update(), prompt_hub.format_http_error(response, base_url)

                data = response.json() if response.content else {}
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if not choices:
                    return gr.update(), "LLM returned no response"

                generated_text = str(choices[0].get("message", {}).get("content", "")).strip()
                if not generated_text:
                    return gr.update(), "LLM returned empty response"

                generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()
                progress(1.0, desc="Done")
                return gr.update(value=generated_text), "Prompt generated successfully"
            except requests.exceptions.ConnectionError:
                if use_local:
                    ollama_url = prompt_hub.normalize_v1_base_url(
                        str(user_config.get("llm_ollama_url", prompt_hub.DEFAULT_OLLAMA_ENDPOINT)),
                        fallback=prompt_hub.DEFAULT_OLLAMA_ENDPOINT,
                    )
                    return gr.update(), f"Could not connect to local Ollama at {ollama_url}. Start `ollama serve` and retry."
                return gr.update(), f"Could not connect to endpoint: {base_url}"
            except requests.exceptions.Timeout:
                return gr.update(), "LLM request timed out (120s)"
            except Exception as e:
                return gr.update(), f"Error: {e}"

        components["llm_use_local_ollama"].change(
            on_use_local_ollama_change,
            inputs=[components["llm_use_local_ollama"], components["llm_endpoint_url"], components["llm_model"]],
            outputs=[components["llm_endpoint_url"], components["llm_model_suggestions"], components["llm_status"], components["llm_model"]],
        )

        components["llm_endpoint_url"].change(
            on_endpoint_change,
            inputs=[components["llm_endpoint_url"]],
            outputs=[components["llm_endpoint_url"]],
        )

        components["refresh_models_btn"].click(
            refresh_models,
            inputs=[components["llm_use_local_ollama"], components["llm_endpoint_url"], components["llm_model"]],
            outputs=[components["llm_model_suggestions"], components["llm_status"], components["llm_model"]],
        )

        components["llm_model_suggestions"].change(
            on_model_suggestion_change,
            inputs=[components["llm_model_suggestions"]],
            outputs=[components["llm_model"], components["llm_status"]],
        )

        components["llm_model"].change(
            on_model_change,
            inputs=[components["llm_model"]],
            outputs=[],
        )

        components["generate_btn"].click(
            generate_with_llm,
            inputs=[
                components["llm_instruction"],
                components["system_prompt"],
                components["llm_model"],
                components["llm_use_local_ollama"],
                components["llm_endpoint_url"],
            ],
            outputs=[components["prompt_text"], components["llm_status"]],
        )

        def send_prompt_to_target(mode, prompt_text, target_id):
            if not prompt_text or not str(prompt_text).strip():
                return "", "Prompt text is empty. Nothing sent."
            enabled_choices = prompt_get_enabled_target_choices(user_config)
            enabled_ids = {value for _label, value in enabled_choices}
            if not target_id or target_id not in enabled_ids:
                return "", "Target is not available or disabled."

            payload = prompt_build_apply_payload(target_id, mode, str(prompt_text), source="prompt_manager")
            target_label = prompt_hub.PROMPT_TARGETS.get(target_id, {}).get("label", target_id)
            return payload, f"Sent to {target_label} ({mode})."

        if prompt_apply_trigger is not None:
            if main_tabs_component is not None:
                def send_and_switch(mode, prompt_text, target_id):
                    payload, status = send_prompt_to_target(mode, prompt_text, target_id)
                    tab_id = prompt_get_target_tab_id(target_id) if payload else ""
                    tab_update = gr.update(selected=tab_id) if tab_id else gr.update()
                    return payload, status, tab_update

                components["send_replace_btn"].click(
                    lambda text, target: send_and_switch("replace", text, target),
                    inputs=[components["prompt_text"], components["pm_target"]],
                    outputs=[prompt_apply_trigger, components["pm_status"], main_tabs_component],
                )
                components["send_append_btn"].click(
                    lambda text, target: send_and_switch("append", text, target),
                    inputs=[components["prompt_text"], components["pm_target"]],
                    outputs=[prompt_apply_trigger, components["pm_status"], main_tabs_component],
                )
            else:
                components["send_replace_btn"].click(
                    lambda text, target: send_prompt_to_target("replace", text, target),
                    inputs=[components["prompt_text"], components["pm_target"]],
                    outputs=[prompt_apply_trigger, components["pm_status"]],
                )
                components["send_append_btn"].click(
                    lambda text, target: send_prompt_to_target("append", text, target),
                    inputs=[components["prompt_text"], components["pm_target"]],
                    outputs=[prompt_apply_trigger, components["pm_status"]],
                )


get_tool_class = lambda: PromptManagerTool


if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone

    run_tool_standalone(PromptManagerTool, port=7875, title="Prompt Manager - Standalone")
