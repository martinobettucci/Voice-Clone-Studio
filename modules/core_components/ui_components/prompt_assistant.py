"""Reusable prompt assistant UI and wiring for generation tabs."""

from __future__ import annotations

from typing import Dict, List

import gradio as gr

import modules.core_components.prompt_hub as prompt_hub


def create_prompt_assistant(
    shared_state: Dict,
    target_ids: List[str],
    default_target_id: str,
    title: str = "Prompt Assistant",
) -> Dict[str, object]:
    """Create a compact prompt assistant block for a tab."""
    choices = prompt_hub.get_enabled_target_choices(shared_state.get("_user_config", {}), target_ids=target_ids)
    if not choices:
        choices = [(prompt_hub.PROMPT_TARGETS[tid]["label"], tid) for tid in target_ids if tid in prompt_hub.PROMPT_TARGETS]

    default_target = default_target_id if default_target_id in [c[1] for c in choices] else choices[0][1]
    default_preset = prompt_hub.get_target_default_preset(default_target)
    saved_names = prompt_hub.get_prompt_names()

    components: Dict[str, object] = {}

    with gr.Accordion(title, open=False):
        components["target"] = gr.Dropdown(
            label="Target Field",
            choices=choices,
            value=default_target,
            interactive=True,
            visible=len(choices) > 1,
        )

        with gr.Row():
            components["saved_prompt"] = gr.Dropdown(
                label="Saved Prompt",
                choices=saved_names,
                value=saved_names[0] if saved_names else None,
                interactive=True,
                scale=3,
            )
            components["refresh_saved"] = gr.Button("Refresh", scale=1)

        with gr.Row():
            components["apply_replace"] = gr.Button("Apply Saved (Replace)", size="sm")
            components["apply_append"] = gr.Button("Apply Saved (Append)", size="sm")

        components["instruction"] = gr.Textbox(
            label="Generate Instruction",
            lines=3,
            placeholder="Describe what text to generate for the selected target field...",
            interactive=True,
        )

        with gr.Row():
            components["system_preset"] = gr.Dropdown(
                label="System Preset",
                choices=prompt_hub.SYSTEM_PROMPT_CHOICES,
                value=default_preset,
                interactive=True,
                scale=1,
            )
            components["custom_system"] = gr.Textbox(
                label="Custom System Override (Optional)",
                lines=2,
                interactive=True,
                scale=2,
                placeholder="Optional: override system prompt for this generation.",
            )

        with gr.Row():
            components["generate_replace"] = gr.Button("Generate + Apply (Replace)", variant="primary")
            components["generate_append"] = gr.Button("Generate + Apply (Append)", variant="primary")

    return components


def wire_prompt_assistant_events(
    assistant: Dict[str, object],
    target_components: Dict[str, object],
    status_component,
    shared_state: Dict,
):
    """Wire assistant controls for local tab apply and generation."""
    user_config = shared_state.get("_user_config", {})

    target_ids = list(target_components.keys())
    target_outputs = [target_components[target_id] for target_id in target_ids]

    def _blank_updates(status_msg=None):
        updates = [gr.update() for _ in target_ids]
        if status_msg is None:
            return (*updates, gr.update())
        return (*updates, status_msg)

    def _resolve_target(selected_target):
        if selected_target in target_components:
            return selected_target
        return target_ids[0]

    def _apply_text(mode, selected_target, text, *current_values):
        target_id = _resolve_target(selected_target)
        if not text or not str(text).strip():
            return _blank_updates("Please provide non-empty text before applying.")

        current_map = dict(zip(target_ids, current_values))
        merged = prompt_hub.merge_text(current_map.get(target_id, ""), str(text), mode)

        updates = []
        for tid in target_ids:
            if tid == target_id:
                updates.append(gr.update(value=merged))
            else:
                updates.append(gr.update())

        label = prompt_hub.PROMPT_TARGETS.get(target_id, {}).get("label", target_id)
        return (*updates, f"Applied to {label} ({mode}).")

    def apply_saved(mode, prompt_name, selected_target, *current_values):
        saved_text = prompt_hub.get_prompt_text(prompt_name)
        if not saved_text:
            return _blank_updates("Select a saved prompt first.")
        return _apply_text(mode, selected_target, saved_text, *current_values)

    def generate_and_apply(mode, instruction, preset_name, custom_system, selected_target, *current_values):
        target_id = _resolve_target(selected_target)
        current_map = dict(zip(target_ids, current_values))
        existing_text = str(current_map.get(target_id, "") or "")
        generated_text, error = prompt_hub.generate_for_target(
            user_config=user_config,
            target_id=target_id,
            instruction=str(instruction or ""),
            preset_override=str(preset_name or ""),
            custom_system_override=str(custom_system or ""),
            existing_text=existing_text,
            apply_mode=mode,
        )
        if error:
            return _blank_updates(error)
        return _apply_text(mode, target_id, generated_text, *current_values)

    def refresh_saved(current_value):
        names = prompt_hub.get_prompt_names()
        value = current_value if current_value in names else (names[0] if names else None)
        return gr.update(choices=names, value=value)

    def on_target_change(selected_target):
        target_id = _resolve_target(selected_target)
        default_preset = prompt_hub.get_target_default_preset(target_id)
        return gr.update(value=default_preset)

    assistant["refresh_saved"].click(
        refresh_saved,
        inputs=[assistant["saved_prompt"]],
        outputs=[assistant["saved_prompt"]],
    )

    assistant["target"].change(
        on_target_change,
        inputs=[assistant["target"]],
        outputs=[assistant["system_preset"]],
    )

    assistant["apply_replace"].click(
        lambda prompt_name, target_id, *vals: apply_saved("replace", prompt_name, target_id, *vals),
        inputs=[assistant["saved_prompt"], assistant["target"], *target_outputs],
        outputs=[*target_outputs, status_component],
    )

    assistant["apply_append"].click(
        lambda prompt_name, target_id, *vals: apply_saved("append", prompt_name, target_id, *vals),
        inputs=[assistant["saved_prompt"], assistant["target"], *target_outputs],
        outputs=[*target_outputs, status_component],
    )

    assistant["generate_replace"].click(
        lambda instruction, preset, custom_system, target_id, *vals: generate_and_apply(
            "replace", instruction, preset, custom_system, target_id, *vals
        ),
        inputs=[
            assistant["instruction"],
            assistant["system_preset"],
            assistant["custom_system"],
            assistant["target"],
            *target_outputs,
        ],
        outputs=[*target_outputs, status_component],
    )

    assistant["generate_append"].click(
        lambda instruction, preset, custom_system, target_id, *vals: generate_and_apply(
            "append", instruction, preset, custom_system, target_id, *vals
        ),
        inputs=[
            assistant["instruction"],
            assistant["system_preset"],
            assistant["custom_system"],
            assistant["target"],
            *target_outputs,
        ],
        outputs=[*target_outputs, status_component],
    )


def wire_prompt_apply_listener(
    prompt_apply_trigger,
    target_components: Dict[str, object],
    status_component,
):
    """Wire cross-tab apply listener for prompt payload trigger."""
    target_ids = list(target_components.keys())
    target_outputs = [target_components[target_id] for target_id in target_ids]

    def on_prompt_apply(trigger_value, *current_values):
        payload = prompt_hub.parse_apply_payload(trigger_value)
        if not payload:
            return (*[gr.update() for _ in target_ids], gr.update())

        target_id = payload.get("target_id", "")
        if target_id not in target_components:
            return (*[gr.update() for _ in target_ids], gr.update())

        incoming = payload.get("text", "")
        if not str(incoming).strip():
            return (*[gr.update() for _ in target_ids], "Prompt text is empty. Nothing applied.")

        mode = payload.get("mode", "replace")
        source = payload.get("source", "external")

        current_map = dict(zip(target_ids, current_values))
        merged = prompt_hub.merge_text(current_map.get(target_id, ""), str(incoming), mode)

        updates = []
        for tid in target_ids:
            if tid == target_id:
                updates.append(gr.update(value=merged))
            else:
                updates.append(gr.update())

        label = prompt_hub.PROMPT_TARGETS.get(target_id, {}).get("label", target_id)
        return (*updates, f"Applied from {source} to {label} ({mode}).")

    prompt_apply_trigger.change(
        on_prompt_apply,
        inputs=[prompt_apply_trigger, *target_outputs],
        outputs=[*target_outputs, status_component],
    )
