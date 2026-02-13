"""Shared prompt utilities for Prompt Manager and generation tabs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

PROMPTS_FILE = Path(__file__).parent.parent.parent / "prompts.json"

DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/v1"

SYSTEM_PROMPTS: Dict[str, str] = {
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
        "Each speaker line MUST start on a new line with their number in brackets, like [1]: or [2]: etc. "
        "Output ONLY the conversation lines in [n]: format."
    ),
    "Voice Style": (
        "You are a voice styling assistant for text-to-speech generation. "
        "Convert user intent into concise style instructions focused on delivery only: tone, pacing, energy, "
        "emotion, clarity, accent, and intensity. "
        "Do not write dialogue or script content. Do not explain. "
        "Output only the final style instruction text."
    ),
    "Sound Design / SFX": (
        "You are a sound design prompt writer. The user will give you a short idea or concept. "
        "Your job is to expand it into a detailed, evocative description of a sound or soundscape. "
        "Focus on texture, layers, timing, spatial qualities, and acoustic characteristics. "
        "Describe what the listener should hear, not see. Output ONLY the final sound description."
    ),
}

SYSTEM_PROMPT_CHOICES = list(SYSTEM_PROMPTS.keys()) + ["Custom"]

PROMPT_TARGETS: Dict[str, Dict[str, str]] = {
    "voice_clone.text": {
        "label": "Voice Clone: Text to Generate",
        "tool": "Voice Clone",
        "tab_id": "tab_voice_clone",
        "component_key": "text_input",
        "default_system_preset": "TTS / Voice",
        "template": (
            "Write final spoken text for voice generation. "
            "Return only the exact text to speak, no extra labels or notes.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "voice_presets.text": {
        "label": "Voice Presets: Text to Generate",
        "tool": "Voice Presets",
        "tab_id": "tab_voice_presets",
        "component_key": "custom_text_input",
        "default_system_preset": "TTS / Voice",
        "template": (
            "Write final spoken text for voice generation. "
            "Return only the exact text to speak, no extra labels or notes.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "voice_presets.style": {
        "label": "Voice Presets: Style Instructions",
        "tool": "Voice Presets",
        "tab_id": "tab_voice_presets",
        "component_key": "custom_instruct_input",
        "default_system_preset": "Voice Style",
        "template": (
            "Generate concise style instructions for TTS. "
            "Return one short instruction phrase only, no quotes, no bullets.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "voice_design.reference": {
        "label": "Voice Design: Reference Text",
        "tool": "Voice Design",
        "tab_id": "tab_voice_design",
        "component_key": "design_text_input",
        "default_system_preset": "TTS / Voice",
        "template": (
            "Write final spoken reference text for a voice design sample. "
            "Return only the text to be spoken.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "voice_design.instructions": {
        "label": "Voice Design: Voice Instructions",
        "tool": "Voice Design",
        "tab_id": "tab_voice_design",
        "component_key": "design_instruct_input",
        "default_system_preset": "Voice Style",
        "template": (
            "Write voice design descriptors only. Focus on timbre, age, energy, pacing, and delivery style. "
            "Return a compact instruction paragraph only.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "conversation.script": {
        "label": "Conversation: Script",
        "tool": "Conversation",
        "tab_id": "tab_conversation",
        "component_key": "conversation_script",
        "default_system_preset": "Conversation",
        "template": (
            "Create a conversation script strictly in [n]: format. "
            "No narration, no stage directions, no markdown.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "sound_effects.prompt": {
        "label": "Sound Effects: Prompt",
        "tool": "Sound Effects",
        "tab_id": "tab_sound_effects",
        "component_key": "sfx_prompt",
        "default_system_preset": "Sound Design / SFX",
        "template": (
            "Write one high-quality sound design prompt for audio generation. "
            "Return only the final prompt text.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "sound_effects.negative": {
        "label": "Sound Effects: Negative Prompt",
        "tool": "Sound Effects",
        "tab_id": "tab_sound_effects",
        "component_key": "sfx_negative_prompt",
        "default_system_preset": "Sound Design / SFX",
        "template": (
            "Write a negative prompt as a comma-separated list of sounds to avoid. "
            "Return only the list, no explanation.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
}


def load_prompts() -> Dict[str, str]:
    """Load prompts from prompts.json."""
    if PROMPTS_FILE.exists():
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_prompts(prompts: Dict[str, str]) -> Dict[str, str]:
    """Save prompts sorted by key."""
    sorted_prompts = dict(sorted(prompts.items(), key=lambda x: x[0].lower()))
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_prompts, f, indent=2, ensure_ascii=False)
    return sorted_prompts


def get_prompt_names() -> List[str]:
    """Return sorted prompt names."""
    return sorted(load_prompts().keys(), key=str.lower)


def get_prompt_text(name: str) -> str:
    """Get saved prompt text by name."""
    if not name:
        return ""
    return load_prompts().get(name, "")


def get_target_config(target_id: str) -> Optional[Dict[str, str]]:
    """Return target config or None."""
    return PROMPT_TARGETS.get(target_id)


def get_target_default_preset(target_id: str) -> str:
    """Return default system preset for target."""
    cfg = get_target_config(target_id)
    if not cfg:
        return SYSTEM_PROMPT_CHOICES[0]
    return cfg.get("default_system_preset", SYSTEM_PROMPT_CHOICES[0])


def get_target_tab_id(target_id: str) -> str:
    """Return destination tab id for target."""
    cfg = get_target_config(target_id)
    return cfg.get("tab_id", "") if cfg else ""


def get_enabled_target_choices(user_config: Dict[str, object], target_ids: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Return enabled prompt target choices as (label, id)."""
    enabled_tools = user_config.get("enabled_tools", {}) if isinstance(user_config, dict) else {}
    choices: List[Tuple[str, str]] = []

    ids = target_ids if target_ids is not None else list(PROMPT_TARGETS.keys())
    for target_id in ids:
        cfg = PROMPT_TARGETS.get(target_id)
        if not cfg:
            continue
        tool_name = cfg.get("tool", "")
        if enabled_tools.get(tool_name, True):
            choices.append((cfg.get("label", target_id), target_id))

    return choices


def normalize_v1_base_url(url: str, fallback: str = DEFAULT_OPENAI_ENDPOINT) -> str:
    """Normalize endpoint base URL to include /v1 exactly once."""
    base = (url or "").strip()
    if not base:
        base = fallback

    base = base.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def get_ollama_tags_url(ollama_v1_url: str) -> str:
    """Convert Ollama /v1 URL to /api/tags URL."""
    v1 = normalize_v1_base_url(ollama_v1_url, fallback=DEFAULT_OLLAMA_ENDPOINT)
    root = v1[:-3] if v1.endswith("/v1") else v1
    return f"{root}/api/tags"


def get_effective_base_url(use_local_ollama: bool, endpoint_url: str, user_config: Dict[str, object]) -> str:
    """Resolve active provider base URL."""
    if use_local_ollama:
        return normalize_v1_base_url(
            str(user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT)),
            fallback=DEFAULT_OLLAMA_ENDPOINT,
        )

    return normalize_v1_base_url(
        endpoint_url or str(user_config.get("llm_endpoint_url", DEFAULT_OPENAI_ENDPOINT)),
        fallback=DEFAULT_OPENAI_ENDPOINT,
    )


def build_headers(user_config: Dict[str, object], use_local_ollama: bool) -> Dict[str, str]:
    """Build HTTP headers for endpoint calls."""
    headers = {"Content-Type": "application/json"}
    api_key = str(user_config.get("llm_api_key", "")).strip()
    if api_key and not use_local_ollama:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_error_message(response: requests.Response) -> str:
    """Extract text from endpoint error payload."""
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


def format_http_error(response: requests.Response, base_url: str) -> str:
    """Map common HTTP statuses to actionable messages."""
    detail = _extract_error_message(response)
    if response.status_code in (401, 403):
        return (
            "Authentication failed (401/403). "
            "Check API key in Settings > LLM Endpoint. "
            f"Endpoint: {base_url}\nDetails: {detail}"
        )
    if response.status_code == 404:
        return (
            "Endpoint or model not found (404). "
            "Check endpoint URL and model name. "
            f"Endpoint: {base_url}\nDetails: {detail}"
        )
    if response.status_code == 400:
        return f"Invalid request (400): {detail}"
    return f"Request failed ({response.status_code}): {detail}"


def _parse_openai_model_ids(payload: object) -> List[str]:
    """Extract model ids from OpenAI-compatible /models response."""
    models: List[str] = []
    if isinstance(payload, dict):
        for item in payload.get("data", []):
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())
    return sorted(set(models), key=lambda x: x.lower())


def discover_available_models(user_config: Dict[str, object], use_local_ollama: bool, endpoint_url: str) -> Tuple[List[str], str]:
    """Discover models from provider."""
    timeout = 12

    if use_local_ollama:
        ollama_v1 = normalize_v1_base_url(
            str(user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT)),
            fallback=DEFAULT_OLLAMA_ENDPOINT,
        )

        tags_url = get_ollama_tags_url(ollama_v1)
        try:
            response = requests.get(tags_url, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                names: List[str] = []
                if isinstance(payload, dict):
                    for item in payload.get("models", []):
                        if isinstance(item, dict):
                            name = item.get("model") or item.get("name")
                            if isinstance(name, str) and name.strip():
                                names.append(name.strip())
                if names:
                    unique = sorted(set(names), key=lambda x: x.lower())
                    return unique, f"Found {len(unique)} local Ollama model(s)."
        except requests.exceptions.ConnectionError:
            return [], f"Could not connect to local Ollama at {ollama_v1}. Start `ollama serve` and retry."
        except requests.exceptions.Timeout:
            return [], f"Timed out while querying local Ollama at {ollama_v1}."
        except Exception:
            pass

        try:
            response = requests.get(f"{ollama_v1}/models", timeout=timeout)
            if response.status_code != 200:
                return [], format_http_error(response, ollama_v1)
            models = _parse_openai_model_ids(response.json())
            if not models:
                return [], "Ollama responded but no models were found. Pull one first, e.g. `ollama pull llama3.1`."
            return models, f"Found {len(models)} local Ollama model(s)."
        except requests.exceptions.ConnectionError:
            return [], f"Could not connect to local Ollama at {ollama_v1}. Start `ollama serve` and retry."
        except requests.exceptions.Timeout:
            return [], f"Timed out while querying local Ollama at {ollama_v1}."
        except Exception as e:
            return [], f"Failed to query local Ollama models: {e}"

    base_url = get_effective_base_url(False, endpoint_url, user_config)
    headers = build_headers(user_config, use_local_ollama=False)
    try:
        response = requests.get(f"{base_url}/models", headers=headers, timeout=timeout)
        if response.status_code != 200:
            return [], format_http_error(response, base_url)
        models = _parse_openai_model_ids(response.json())
        if not models:
            return [], "No models returned by /models. You can still type the model name manually."
        return models, f"Found {len(models)} model(s) from endpoint."
    except requests.exceptions.ConnectionError:
        return [], f"Could not connect to endpoint: {base_url}."
    except requests.exceptions.Timeout:
        return [], f"Timed out while querying models from {base_url}."
    except Exception as e:
        return [], f"Failed to query models: {e}"


def _build_target_instruction(target_id: str, instruction: str) -> str:
    """Build user message with target-specific template."""
    cfg = get_target_config(target_id)
    if not cfg:
        return instruction
    template = cfg.get("template", "{instruction}")
    return template.format(instruction=instruction.strip())


def generate_for_target(
    user_config: Dict[str, object],
    target_id: str,
    instruction: str,
    preset_override: Optional[str] = None,
    custom_system_override: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Generate text tailored for target field using global endpoint settings.

    Returns:
        (generated_text, error_message)
    """
    if not instruction or not instruction.strip():
        return "", "Please enter instructions."

    cfg = get_target_config(target_id)
    if not cfg:
        return "", f"Unknown prompt target: {target_id}"

    use_local = bool(user_config.get("llm_use_local_ollama", False))
    endpoint_url = str(user_config.get("llm_endpoint_url", DEFAULT_OPENAI_ENDPOINT))
    base_url = get_effective_base_url(use_local, endpoint_url, user_config)

    model = str(user_config.get("llm_model", "gpt-4o-mini")).strip() or "gpt-4o-mini"

    default_preset = cfg.get("default_system_preset", SYSTEM_PROMPT_CHOICES[0])
    chosen_preset = preset_override if preset_override in SYSTEM_PROMPTS else default_preset
    custom_system = (custom_system_override or "").strip()
    effective_system = custom_system if custom_system else SYSTEM_PROMPTS.get(chosen_preset, SYSTEM_PROMPTS[SYSTEM_PROMPT_CHOICES[0]])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": _build_target_instruction(target_id, instruction)},
        ],
        "stream": False,
        "temperature": 0.8,
        "top_p": 0.95,
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=build_headers(user_config, use_local),
            timeout=120,
        )
        if response.status_code != 200:
            return "", format_http_error(response, base_url)

        data = response.json() if response.content else {}
        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            return "", "LLM returned no response."

        generated = str(choices[0].get("message", {}).get("content", "")).strip()
        if not generated:
            return "", "LLM returned empty response."

        generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.DOTALL).strip()
        return generated, None
    except requests.exceptions.ConnectionError:
        if use_local:
            ollama = normalize_v1_base_url(
                str(user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT)),
                fallback=DEFAULT_OLLAMA_ENDPOINT,
            )
            return "", f"Could not connect to local Ollama at {ollama}. Start `ollama serve` and retry."
        return "", f"Could not connect to endpoint: {base_url}"
    except requests.exceptions.Timeout:
        return "", "LLM request timed out (120s)."
    except Exception as e:
        return "", f"Error: {e}"


def merge_text(existing: str, new_text: str, mode: str) -> str:
    """Merge text using replace/append semantics."""
    old = existing or ""
    new = new_text or ""
    if mode == "append":
        if not old.strip():
            return new
        if not new.strip():
            return old
        return f"{old.rstrip()}\n{new.lstrip()}"
    return new


def build_apply_payload(target_id: str, mode: str, text: str, source: str = "prompt_assistant") -> str:
    """Create serialized prompt-apply payload with unique nonce."""
    payload = {
        "target_id": target_id,
        "mode": mode,
        "text": text,
        "source": source,
        "nonce": time.time_ns(),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_apply_payload(raw_value: str) -> Optional[Dict[str, str]]:
    """Parse and validate serialized apply payload."""
    if not raw_value or not str(raw_value).strip():
        return None

    try:
        payload = json.loads(raw_value)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    target_id = str(payload.get("target_id", "")).strip()
    mode = str(payload.get("mode", "")).strip().lower()
    text = str(payload.get("text", ""))
    source = str(payload.get("source", "")).strip() or "unknown"

    if target_id not in PROMPT_TARGETS:
        return None
    if mode not in {"replace", "append"}:
        return None

    return {
        "target_id": target_id,
        "mode": mode,
        "text": text,
        "source": source,
    }
