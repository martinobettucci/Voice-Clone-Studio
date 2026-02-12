"""
Prompt Manager Tool

Save, browse, and generate text prompts for TTS and sound effects.
Uses OpenAI-compatible chat completion endpoints with optional local Ollama mode.
Prompts are stored in a standalone prompts.json at the project root.
"""

import json
import re
from pathlib import Path

import gradio as gr
import requests

from modules.core_components.tool_base import Tool, ToolConfig

# ============================================================================
# Prompts file (standalone, same level as config.json)
# ============================================================================
PROMPTS_FILE = Path(__file__).parent.parent.parent.parent / "prompts.json"

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/v1"

# System prompt presets
SYSTEM_PROMPTS = {
    "TTS / Voice": (
        "You are a script writer for voice acting. The user will give you a short idea or concept. "
        "Your job is to write dialogue or monologue in FIRST PERSON, as if the speaker is saying it aloud. "
        "Never describe the speaker in third person. Write the actual words they would speak. "
        "Focus on tone, emotion, pacing, and natural speech patterns. "
        "Output ONLY the spoken text, nothing else - no stage directions, no quotation marks."
    ),
    "Conversation": (
        "You are a script writer for multi-speaker conversations. The user will give you a topic, scenario, "
        "or concept along with the number of speakers to use. "
        "Your job is to write a natural conversation where each speaker talks in FIRST PERSON to the others. "
        "Each speaker's line MUST start on a new line with their number in brackets, like [1]: or [2]: etc. "
        "Example format:\n"
        "[1]: Hey, did you hear about the new project?\n"
        "[2]: Yeah, I just got the email this morning.\n"
        "[1]: What do you think about the timeline?\n"
        "Write natural, flowing dialogue with realistic turn-taking. "
        "Vary sentence length and speaking style between speakers to give each a distinct voice. "
        "Output ONLY the conversation lines in the [n]: format, nothing else - no narration, no stage directions, no quotation marks."
    ),
    "Sound Design / SFX": (
        "You are a sound design prompt writer. The user will give you a short idea or concept. "
        "Your job is to expand it into a detailed, evocative description of a sound or soundscape. "
        "Focus on texture, layers, timing, spatial qualities, and acoustic characteristics. "
        "Describe what the listener should hear, not see. "
        "Output ONLY the final sound description, nothing else."
    ),
}

SYSTEM_PROMPT_CHOICES = list(SYSTEM_PROMPTS.keys()) + ["Custom"]


# ============================================================================
# Endpoint helpers
# ============================================================================

def _normalize_v1_base_url(url: str, fallback: str = DEFAULT_OPENAI_ENDPOINT) -> str:
    """Normalize endpoint base URL to include /v1 exactly once."""
    base = (url or "").strip()
    if not base:
        base = fallback

    base = base.rstrip("/")

    # If user pasted full completion path, normalize back to /v1
    for suffix in ("/chat/completions", "/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _get_ollama_tags_url(ollama_v1_url: str) -> str:
    """Convert Ollama v1 URL to Ollama native tags endpoint URL."""
    v1 = _normalize_v1_base_url(ollama_v1_url, fallback=DEFAULT_OLLAMA_ENDPOINT)
    root = v1[:-3] if v1.endswith("/v1") else v1
    return f"{root}/api/tags"


def _get_effective_base_url(use_local_ollama: bool, endpoint_url: str, user_config: dict) -> str:
    """Get the active endpoint base URL."""
    if use_local_ollama:
        return _normalize_v1_base_url(
            user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT),
            fallback=DEFAULT_OLLAMA_ENDPOINT,
        )

    return _normalize_v1_base_url(
        endpoint_url or user_config.get("llm_endpoint_url", DEFAULT_OPENAI_ENDPOINT),
        fallback=DEFAULT_OPENAI_ENDPOINT,
    )


def _build_headers(user_config: dict, use_local_ollama: bool) -> dict:
    """Build request headers with optional bearer auth."""
    headers = {"Content-Type": "application/json"}
    api_key = (user_config.get("llm_api_key") or "").strip()
    if api_key and not use_local_ollama:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_error_message(response: requests.Response) -> str:
    """Extract best-effort message from endpoint error payload."""
    try:
        payload = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:300] if text else f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("error")
            if msg:
                return str(msg)
        if isinstance(err, str):
            return err
        msg = payload.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()

    return f"HTTP {response.status_code}"


def _format_http_error(response: requests.Response, base_url: str) -> str:
    """Map common HTTP statuses to actionable messages."""
    detail = _extract_error_message(response)
    if response.status_code in (401, 403):
        return (
            "Authentication failed (401/403). "
            "Check your API key in Settings > LLM Endpoint, then retry. "
            f"Endpoint: {base_url}\nDetails: {detail}"
        )
    if response.status_code == 404:
        return (
            "Endpoint or model not found (404). "
            "Verify endpoint URL includes the correct /v1 API and model name is valid. "
            f"Endpoint: {base_url}\nDetails: {detail}"
        )
    if response.status_code == 400:
        return f"Invalid request (400): {detail}"
    return f"Request failed ({response.status_code}): {detail}"


def _parse_openai_model_ids(payload: dict) -> list[str]:
    """Extract model IDs from OpenAI-compatible /models response."""
    models = []
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id.strip():
                models.append(model_id.strip())
    # De-duplicate while preserving case, sort case-insensitive for stable UI
    unique = sorted(set(models), key=lambda x: x.lower())
    return unique


def _discover_available_models(user_config: dict, use_local_ollama: bool, endpoint_url: str) -> tuple[list[str], str]:
    """Discover available models from active provider."""
    timeout = 12

    if use_local_ollama:
        ollama_v1 = _normalize_v1_base_url(
            user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT),
            fallback=DEFAULT_OLLAMA_ENDPOINT,
        )

        # Preferred Ollama API
        tags_url = _get_ollama_tags_url(ollama_v1)
        try:
            response = requests.get(tags_url, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                models = []
                for item in payload.get("models", []) if isinstance(payload, dict) else []:
                    if isinstance(item, dict):
                        name = item.get("model") or item.get("name")
                        if isinstance(name, str) and name.strip():
                            models.append(name.strip())
                if models:
                    deduped = sorted(set(models), key=lambda x: x.lower())
                    return deduped, f"Found {len(deduped)} local Ollama model(s)."
        except requests.exceptions.ConnectionError:
            return [], (
                f"Could not connect to local Ollama at {ollama_v1}. "
                "Start Ollama with `ollama serve` and retry."
            )
        except requests.exceptions.Timeout:
            return [], f"Timed out while querying local Ollama at {ollama_v1}."
        except Exception:
            # Fall through to OpenAI-compatible /models fallback
            pass

        # Fallback: OpenAI-compatible Ollama endpoint
        try:
            response = requests.get(f"{ollama_v1}/models", timeout=timeout)
            if response.status_code != 200:
                return [], _format_http_error(response, ollama_v1)

            models = _parse_openai_model_ids(response.json())
            if not models:
                return [], (
                    "Ollama responded but no models were discovered. "
                    "Pull a model first, e.g. `ollama pull llama3.1`."
                )
            return models, f"Found {len(models)} local Ollama model(s)."
        except requests.exceptions.ConnectionError:
            return [], (
                f"Could not connect to local Ollama at {ollama_v1}. "
                "Start Ollama with `ollama serve` and retry."
            )
        except requests.exceptions.Timeout:
            return [], f"Timed out while querying local Ollama at {ollama_v1}."
        except Exception as e:
            return [], f"Failed to query local Ollama models: {e}"

    base_url = _get_effective_base_url(False, endpoint_url, user_config)
    headers = _build_headers(user_config, use_local_ollama=False)

    try:
        response = requests.get(f"{base_url}/models", headers=headers, timeout=timeout)
        if response.status_code != 200:
            return [], _format_http_error(response, base_url)

        models = _parse_openai_model_ids(response.json())
        if not models:
            return [], (
                "No models returned by endpoint /models. "
                "If this provider does not implement /models, type the model name manually."
            )
        return models, f"Found {len(models)} model(s) from endpoint."
    except requests.exceptions.ConnectionError:
        return [], (
            f"Could not connect to endpoint: {base_url}. "
            "Check URL/network and retry."
        )
    except requests.exceptions.Timeout:
        return [], f"Timed out while querying models from {base_url}."
    except Exception as e:
        return [], f"Failed to query models: {e}"


# ============================================================================
# Prompts file management
# ============================================================================

def _load_prompts():
    """Load prompts from prompts.json.

    Returns:
        Dictionary of name -> text pairs
    """
    if PROMPTS_FILE.exists():
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_prompts(prompts):
    """Save prompts to prompts.json, sorted alphabetically.

    Args:
        prompts: Dictionary of name -> text pairs

    Returns:
        Sorted prompts dictionary
    """
    sorted_prompts = dict(sorted(prompts.items(), key=lambda x: x[0].lower()))
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_prompts, f, indent=2, ensure_ascii=False)
    return sorted_prompts


def _get_prompt_names():
    """Get list of prompt names for FileLister."""
    prompts = _load_prompts()
    return sorted(prompts.keys(), key=str.lower)


# ============================================================================
# Prompt Manager Tool
# ============================================================================

class PromptManagerTool(Tool):
    """Prompt Manager - save, browse, and generate prompts with an LLM."""

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
        saved_endpoint = _normalize_v1_base_url(
            user_config.get("llm_endpoint_url", DEFAULT_OPENAI_ENDPOINT),
            fallback=DEFAULT_OPENAI_ENDPOINT,
        )
        ollama_endpoint = _normalize_v1_base_url(
            user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT),
            fallback=DEFAULT_OLLAMA_ENDPOINT,
        )
        displayed_endpoint = ollama_endpoint if use_local_ollama else saved_endpoint

        saved_llm_model = (user_config.get("llm_model") or "").strip() or "gpt-4o-mini"

        components = {}

        with gr.TabItem("Prompt Manager"):
            with gr.Row():
                # Left column: prompt editor and saved prompts
                with gr.Column(scale=2):
                    gr.Markdown("### Prompt Result")

                    components["prompt_text"] = gr.Textbox(
                        label="Prompt",
                        lines=8,
                        placeholder="Write your prompt here, or select a saved one from the list...",
                        interactive=True,
                    )

                    with gr.Row():
                        components["save_btn"] = gr.Button(
                            "Save Prompt",
                            variant="primary",
                            scale=1,
                        )
                        components["delete_btn"] = gr.Button(
                            "Delete",
                            variant="stop",
                            scale=1,
                        )
                        components["clear_btn"] = gr.Button("Clear", scale=1)

                    components["pm_status"] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                    )

                    gr.Markdown("### Saved Prompts")
                    components["prompt_lister"] = FileLister(
                        value=_get_prompt_names(),
                        height=250,
                        show_footer=False,
                        interactive=True,
                    )

                # Right column: LLM prompt generation
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
                        choices=SYSTEM_PROMPT_CHOICES,
                        value=SYSTEM_PROMPT_CHOICES[0],
                        interactive=True,
                    )

                    components["system_prompt"] = gr.Textbox(
                        label="System Prompt",
                        lines=4,
                        value=SYSTEM_PROMPTS[SYSTEM_PROMPT_CHOICES[0]],
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
                        components["refresh_models_btn"] = gr.Button(
                            "Refresh Models",
                            scale=1,
                        )

                    components["llm_model_suggestions"] = gr.Dropdown(
                        label="Model Suggestions",
                        choices=[],
                        value=None,
                        allow_custom_value=False,
                        interactive=True,
                    )

                    components["generate_btn"] = gr.Button(
                        "Generate Prompt",
                        variant="primary",
                    )

                    components["llm_status"] = gr.Textbox(
                        label="LLM Status",
                        interactive=False,
                        lines=2,
                    )

            # Hidden state for prompt save modal
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

        # --- Select prompt from lister ---
        def load_selected_prompt(lister_value):
            """Load a selected prompt into the text box."""
            if not lister_value:
                return gr.update()
            selected = lister_value.get("selected", [])
            if len(selected) != 1:
                return gr.update()

            prompt_name = selected[0]
            prompts = _load_prompts()
            text = prompts.get(prompt_name, "")
            return gr.update(value=text)

        components["prompt_lister"].change(
            load_selected_prompt,
            inputs=[components["prompt_lister"]],
            outputs=[components["prompt_text"]],
        )

        # --- Clear button ---
        components["clear_btn"].click(
            fn=lambda: (gr.update(value=""), ""),
            outputs=[components["prompt_text"], components["pm_status"]],
        )

        # --- Save button: open input modal ---
        save_modal_js = show_input_modal_js(
            title="Save Prompt",
            message="Enter a name for this prompt:",
            placeholder="e.g., Dramatic Pirate, Thunder Storm, Calm Narrator",
            context="save_prompt_",
        )

        def get_existing_prompt_names():
            """Return JSON list of existing prompt names for overwrite detection."""
            names = _get_prompt_names()
            return json.dumps(names)

        def get_selected_name(lister_value):
            """Get currently selected prompt name as suggested default."""
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

        # Step 1: Get existing names, then step 2: open modal
        components["save_btn"].click(
            fn=lambda lister_val: (get_existing_prompt_names(), get_selected_name(lister_val)),
            inputs=[components["prompt_lister"]],
            outputs=[components["pm_existing_files_json"], components["pm_suggested_name"]],
        ).then(
            fn=None,
            inputs=[components["pm_existing_files_json"], components["pm_suggested_name"]],
            js=save_js,
        )

        # --- Input modal handler: save prompt ---
        def handle_save_prompt(input_value, prompt_text):
            """Process input modal result for saving prompts."""
            no_update = gr.update(), gr.update()

            if not input_value or not input_value.startswith("save_prompt_"):
                return no_update

            # Check for cancel
            parts = input_value.split("_", 3)
            if len(parts) >= 3 and parts[2] == "cancel":
                return no_update

            # Extract name: "save_prompt_<name>_<uuid>"
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

                new_names = _get_prompt_names()
                return f"Saved: {chosen_name.strip()}", gr.update(value=new_names)
            except Exception as e:
                return f"Error saving: {e}", gr.update()

        input_trigger.change(
            handle_save_prompt,
            inputs=[input_trigger, components["prompt_text"]],
            outputs=[components["pm_status"], components["prompt_lister"]],
        )

        # --- Delete button: confirm and delete ---
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
            """Delete the selected prompt after confirmation."""
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

                new_names = _get_prompt_names()
                return (
                    f"Deleted: {prompt_name}",
                    gr.update(value=new_names),
                    gr.update(value=""),
                )
            except Exception as e:
                return f"Error deleting: {e}", gr.update(), gr.update()

        confirm_trigger.change(
            handle_delete_prompt,
            inputs=[confirm_trigger, components["prompt_lister"]],
            outputs=[components["pm_status"], components["prompt_lister"], components["prompt_text"]],
        )

        # --- System prompt preset selector ---
        def on_preset_change(preset_name):
            if preset_name in SYSTEM_PROMPTS:
                return gr.update(value=SYSTEM_PROMPTS[preset_name])
            return gr.update()  # Custom - leave text as-is

        components["system_prompt_preset"].change(
            on_preset_change,
            inputs=[components["system_prompt_preset"]],
            outputs=[components["system_prompt"]],
        )

        def refresh_models(use_local_ollama, endpoint_url, current_model):
            """Refresh model suggestions from active provider."""
            models, status = _discover_available_models(user_config, bool(use_local_ollama), endpoint_url)
            if not models:
                return gr.update(choices=[], value=None), status, gr.update()

            selected = current_model.strip() if current_model and current_model.strip() else models[0]
            selected_dropdown = selected if selected in models else models[0]
            model_update = gr.update()
            if not current_model or not current_model.strip():
                model_update = gr.update(value=models[0])
                save_preference("llm_model", models[0])

            return (
                gr.update(choices=models, value=selected_dropdown),
                status,
                model_update,
            )

        def on_use_local_ollama_change(use_local_ollama, endpoint_url, current_model):
            """Toggle local Ollama mode and refresh endpoint/model state."""
            use_local = bool(use_local_ollama)
            save_preference("llm_use_local_ollama", use_local)

            if use_local:
                endpoint_value = _normalize_v1_base_url(
                    user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT),
                    fallback=DEFAULT_OLLAMA_ENDPOINT,
                )
            else:
                endpoint_value = _normalize_v1_base_url(
                    user_config.get("llm_endpoint_url", endpoint_url or DEFAULT_OPENAI_ENDPOINT),
                    fallback=DEFAULT_OPENAI_ENDPOINT,
                )

            suggestions_update, status, model_update = refresh_models(use_local, endpoint_value, current_model)
            endpoint_update = gr.update(value=endpoint_value, interactive=not use_local)
            return endpoint_update, suggestions_update, status, model_update

        def on_endpoint_change(endpoint_url):
            """Normalize and persist endpoint URL."""
            normalized = _normalize_v1_base_url(endpoint_url, fallback=DEFAULT_OPENAI_ENDPOINT)
            save_preference("llm_endpoint_url", normalized)
            return gr.update(value=normalized)

        def on_model_change(model_name):
            model = (model_name or "").strip()
            if model:
                save_preference("llm_model", model)

        def on_model_suggestion_change(suggested_model):
            """Apply a suggested model to the authoritative model textbox."""
            if not suggested_model:
                return gr.update(), gr.update()
            save_preference("llm_model", suggested_model)
            return gr.update(value=suggested_model), f"Selected model: {suggested_model}"

        # --- Generate prompt with OpenAI-compatible endpoint ---
        def generate_with_llm(instruction, system_prompt, model_name, use_local_ollama, endpoint_url, progress=gr.Progress()):
            """Send instruction to LLM endpoint and return generated prompt."""
            if not instruction or not instruction.strip():
                return gr.update(), "Please enter instructions for the LLM"

            model = (model_name or "").strip()
            if not model:
                return gr.update(), "Please enter a model name"

            use_local = bool(use_local_ollama)
            base_url = _get_effective_base_url(use_local, endpoint_url, user_config)

            if not use_local:
                save_preference("llm_endpoint_url", base_url)
            save_preference("llm_model", model)

            try:
                progress(0.2, desc="Sending request...")

                effective_system = (
                    system_prompt.strip()
                    if system_prompt and system_prompt.strip()
                    else SYSTEM_PROMPTS[SYSTEM_PROMPT_CHOICES[0]]
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
                    headers=_build_headers(user_config, use_local),
                    timeout=120,
                )

                if response.status_code != 200:
                    return gr.update(), _format_http_error(response, base_url)

                data = response.json()
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if not choices:
                    return gr.update(), "LLM returned no response"

                generated_text = choices[0].get("message", {}).get("content", "").strip()
                if not generated_text:
                    return gr.update(), "LLM returned empty response"

                # Remove hidden thinking output where providers include it inline
                generated_text = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()

                progress(1.0, desc="Done")
                return gr.update(value=generated_text), "Prompt generated successfully"

            except requests.exceptions.ConnectionError:
                if use_local:
                    ollama_url = _normalize_v1_base_url(
                        user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT),
                        fallback=DEFAULT_OLLAMA_ENDPOINT,
                    )
                    return (
                        gr.update(),
                        f"Could not connect to local Ollama at {ollama_url}. Start `ollama serve` and retry.",
                    )
                return gr.update(), f"Could not connect to endpoint: {base_url}"
            except requests.exceptions.Timeout:
                return gr.update(), "LLM request timed out (120s)"
            except Exception as e:
                return gr.update(), f"Error: {e}"

        components["llm_use_local_ollama"].change(
            on_use_local_ollama_change,
            inputs=[
                components["llm_use_local_ollama"],
                components["llm_endpoint_url"],
                components["llm_model"],
            ],
            outputs=[
                components["llm_endpoint_url"],
                components["llm_model_suggestions"],
                components["llm_status"],
                components["llm_model"],
            ],
        )

        components["llm_endpoint_url"].change(
            on_endpoint_change,
            inputs=[components["llm_endpoint_url"]],
            outputs=[components["llm_endpoint_url"]],
        )

        components["refresh_models_btn"].click(
            refresh_models,
            inputs=[
                components["llm_use_local_ollama"],
                components["llm_endpoint_url"],
                components["llm_model"],
            ],
            outputs=[
                components["llm_model_suggestions"],
                components["llm_status"],
                components["llm_model"],
            ],
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
            outputs=[
                components["prompt_text"],
                components["llm_status"],
            ],
        )


# Export for registry
get_tool_class = lambda: PromptManagerTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone

    run_tool_standalone(PromptManagerTool, port=7875, title="Prompt Manager - Standalone")
