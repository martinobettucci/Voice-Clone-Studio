"""
Prompt Manager Tool

Save, browse, and generate text prompts for TTS and sound effects.
Includes an LLM-powered prompt generator using llama.cpp.
Prompts are stored in a standalone prompts.json at the project root.
"""

import json
import os
import sys
import subprocess
import signal
import atexit
import time
import ctypes
import requests
import gradio as gr
from pathlib import Path

from modules.core_components.tool_base import Tool, ToolConfig

# ============================================================================
# Prompts file (standalone, same level as config.json)
# ============================================================================
PROMPTS_FILE = Path(__file__).parent.parent.parent.parent / "prompts.json"

# ============================================================================
# LLM Models available for download
# ============================================================================
LLM_MODELS = {
    "Qwen3-4B-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-4B-GGUF",
    },
    "Qwen3-8B-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-8B-GGUF",
    },
}

# System prompt presets
SYSTEM_PROMPTS = {
    "TTS / Voice": (
        "You are a script writer for voice acting. The user will give you a short idea or concept. "
        "Your job is to write dialogue or monologue in FIRST PERSON, as if the speaker is saying it aloud. "
        "Never describe the speaker in third person. Write the actual words they would speak. "
        "Focus on tone, emotion, pacing, and natural speech patterns. "
        "Output ONLY the spoken text, nothing else — no stage directions, no quotation marks."
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
# llama.cpp server management
# ============================================================================
_server_process = None
_current_model = None
_job_handle = None
SERVER_PORT = 8099  # Use a different port to avoid conflicts


def _setup_windows_job_object():
    """Create a Windows Job Object that kills child processes when parent exits."""
    global _job_handle
    if os.name != 'nt' or _job_handle:
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        extended_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        extended_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        if not kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(extended_info), ctypes.sizeof(extended_info)
        ):
            kernel32.CloseHandle(job)
            raise ctypes.WinError(ctypes.get_last_error())

        _job_handle = job
    except Exception:
        pass


def _assign_process_to_job(pid):
    """Assign subprocess pid to job object so it gets killed when parent exits."""
    global _job_handle
    if os.name != 'nt' or not _job_handle or not pid:
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        PROCESS_ALL_ACCESS = 0x1F0FFF
        proc_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, int(pid))
        if not proc_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(_job_handle, proc_handle):
            kernel32.CloseHandle(proc_handle)
            raise ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(proc_handle)
    except Exception:
        pass


def _cleanup_server():
    """Cleanup function to stop server on exit."""
    global _server_process, _job_handle
    if _server_process:
        try:
            _server_process.terminate()
            _server_process.wait(timeout=5)
        except Exception:
            try:
                _server_process.kill()
            except Exception:
                pass
        _server_process = None

    if os.name == 'nt' and _job_handle:
        try:
            ctypes.WinDLL('kernel32').CloseHandle(_job_handle)
        except Exception:
            pass
        _job_handle = None


# Register cleanup
atexit.register(_cleanup_server)


def _is_server_alive():
    """Check if llama.cpp server is responding."""
    try:
        r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _kill_llama_on_port():
    """Kill any process listening on our server port (safety net for orphans)."""
    try:
        if os.name == 'nt':
            # Parse netstat for PIDs on our port
            result = subprocess.run(
                ["netstat", "-aon"],
                capture_output=True, text=True, timeout=10
            )
            pids = set()
            for line in result.stdout.splitlines():
                if f":{SERVER_PORT}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        try:
                            pid = int(parts[-1])
                            if pid > 0:
                                pids.add(pid)
                        except ValueError:
                            pass
            for pid in pids:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass
        else:
            subprocess.run(
                f"lsof -ti:{SERVER_PORT} | xargs -r kill -9",
                shell=True, capture_output=True, timeout=10
            )
    except Exception:
        pass


def _get_llama_models_dir(user_config):
    """Get the path to the default models/llama/ directory."""
    project_root = PROMPTS_FILE.parent
    models_base = project_root / user_config.get("models_folder", "models")
    llama_dir = models_base / "llama"
    llama_dir.mkdir(parents=True, exist_ok=True)
    return llama_dir


def _get_user_llama_models_dir(user_config):
    """Get the user-specified llama models directory, if configured.

    Returns:
        Path or None if not configured / empty.
    """
    user_path = user_config.get("llama_models_path", "").strip()
    if not user_path:
        return None
    p = Path(user_path)
    if p.exists() and p.is_dir():
        return p
    # Try creating it
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return None


def _get_local_gguf_models(user_config):
    """Get list of local .gguf model filenames from all search directories."""
    seen = set()
    models = []

    # 1. Default models/llama/ folder
    llama_dir = _get_llama_models_dir(user_config)
    if llama_dir.exists():
        for f in sorted(llama_dir.glob("*.gguf")):
            if f.name not in seen:
                seen.add(f.name)
                models.append(f.name)

    # 2. User-specified models folder
    user_dir = _get_user_llama_models_dir(user_config)
    if user_dir and user_dir.exists():
        for f in sorted(user_dir.glob("*.gguf")):
            if f.name not in seen:
                seen.add(f.name)
                models.append(f.name)

    return models


def _get_all_model_choices(user_config):
    """Get combined list: local models first, then downloadable ones not yet local."""
    local = _get_local_gguf_models(user_config)
    all_choices = list(local)
    for name in LLM_MODELS:
        if name not in all_choices:
            all_choices.append(name)
    return all_choices


def _find_model_path(model_name, user_config):
    """Find the full path to a local model, searching all directories.

    Returns:
        Path string or None if not found.
    """
    # Check default models/llama/ first
    llama_dir = _get_llama_models_dir(user_config)
    default_path = llama_dir / model_name
    if default_path.exists():
        return str(default_path)

    # Check user-specified directory
    user_dir = _get_user_llama_models_dir(user_config)
    if user_dir:
        user_path = user_dir / model_name
        if user_path.exists():
            return str(user_path)

    return None


def _is_model_local(model_name, user_config):
    """Check if a model exists locally in any search directory."""
    return _find_model_path(model_name, user_config) is not None


def _download_model(model_name, user_config, progress=None):
    """Download a model from HuggingFace.

    Args:
        model_name: Filename of the model
        user_config: User config dict
        progress: Optional Gradio progress callback

    Returns:
        Path to downloaded model or None on error
    """
    if model_name not in LLM_MODELS:
        return None

    model_info = LLM_MODELS[model_name]
    repo_id = model_info["repo"]

    # Download to user-specified directory if configured, otherwise default
    user_dir = _get_user_llama_models_dir(user_config)
    download_dir = user_dir if user_dir else _get_llama_models_dir(user_config)
    local_path = download_dir / model_name

    if local_path.exists():
        return str(local_path)

    try:
        from huggingface_hub import HfApi
        if progress:
            progress(0.0, desc=f"Downloading {model_name}...")

        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
        file_metadata = next((f for f in repo_info.siblings if f.rfilename == model_name), None)
        if not file_metadata or file_metadata.size is None:
            return None

        total_size = file_metadata.size
        download_url = f"https://huggingface.co/{repo_id}/resolve/main/{model_name}"

        downloaded = 0
        with requests.get(download_url, stream=True) as r, open(str(local_path), "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress and total_size > 0:
                        pct = downloaded / total_size
                        progress(pct, desc=f"Downloading {model_name}... {downloaded // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB")

        if progress:
            progress(1.0, desc="Download complete")
        return str(local_path)
    except Exception as e:
        # Clean up partial download
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass
        print(f"[Prompt Manager] Error downloading {model_name}: {e}")
        return None


def _start_server(model_name, user_config, progress=None):
    """Start llama.cpp server with specified model.

    Returns:
        Tuple: (success, error_message)
    """
    global _server_process, _current_model

    # If already running with same model, reuse
    if _server_process and _current_model == model_name and _is_server_alive():
        return True, None

    # Stop any existing server
    _stop_server()

    # Unload all AI models to free VRAM before starting llama.cpp
    try:
        from modules.core_components.ai_models.tts_manager import get_tts_manager
        from modules.core_components.ai_models.asr_manager import get_asr_manager
        from modules.core_components.ai_models.foley_manager import get_foley_manager

        tts = get_tts_manager()
        if tts:
            tts.unload_all()
        asr = get_asr_manager()
        if asr:
            asr.unload_all()
        foley = get_foley_manager()
        if foley:
            foley.unload_all()
    except Exception:
        pass

    # Download if not local
    if not _is_model_local(model_name, user_config):
        if model_name in LLM_MODELS:
            if progress:
                progress(0.1, desc=f"Downloading {model_name}...")
            model_path = _download_model(model_name, user_config, progress)
            if not model_path:
                return False, f"Failed to download model: {model_name}"
        else:
            return False, f"Model not found: {model_name}"
    else:
        model_path = _find_model_path(model_name, user_config)

    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"

    try:
        if progress:
            progress(0.3, desc="Starting llama.cpp server...")

        # Determine executable — use custom path from config if set
        custom_llama_path = user_config.get("llama_cpp_path", "").strip()
        if os.name == 'nt':
            if custom_llama_path:
                server_cmd = str(Path(custom_llama_path) / "llama-server.exe")
            else:
                server_cmd = "llama-server.exe"
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            if custom_llama_path:
                server_cmd = str(Path(custom_llama_path) / "llama-server")
            else:
                server_cmd = "llama-server"
            creation_flags = 0

        cmd_args = [
            server_cmd, "-m", model_path,
            "--port", str(SERVER_PORT),
            "--no-warmup",
            "-c", "4096"
        ]

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }

        if os.name == 'nt':
            popen_kwargs["creationflags"] = creation_flags
        else:
            # On Unix, set PR_SET_PDEATHSIG so child gets SIGTERM when parent dies
            def _set_pdeathsig():
                try:
                    for libname in ("libc.so.6", "libc.dylib", "libc.so"):
                        try:
                            libc = ctypes.CDLL(libname)
                            break
                        except Exception:
                            libc = None
                    if not libc:
                        return
                    PR_SET_PDEATHSIG = 1
                    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                except Exception:
                    return
            popen_kwargs["preexec_fn"] = _set_pdeathsig

        _server_process = subprocess.Popen(cmd_args, **popen_kwargs)

        # On Windows, attach to job object
        if os.name == 'nt':
            _setup_windows_job_object()
            _assign_process_to_job(_server_process.pid)

        _current_model = model_name

        # Wait for server to be ready (up to 60 seconds for large models)
        for i in range(60):
            time.sleep(1)
            if progress:
                progress(0.3 + (i / 60) * 0.3, desc=f"Waiting for server... ({i + 1}s)")
            if _is_server_alive():
                return True, None
            # Check if process crashed
            if _server_process.poll() is not None:
                stderr = _server_process.stderr.read().decode(errors='replace') if _server_process.stderr else ""
                return False, f"Server exited unexpectedly.\n{stderr[:500]}"

        _stop_server()
        return False, "Server did not start in time (60s timeout)"

    except FileNotFoundError:
        custom_hint = ""
        if not user_config.get("llama_cpp_path", "").strip():
            custom_hint = "\nOr set the llama.cpp path in Settings > Folder Paths."
        return False, (
            "llama-server not found. Please install llama.cpp and add to PATH."
            f"{custom_hint}\n"
            "Installation: https://github.com/ggml-org/llama.cpp"
        )
    except Exception as e:
        return False, f"Error starting server: {e}"


def _stop_server():
    """Stop the llama.cpp server and ensure VRAM is freed."""
    global _server_process, _current_model
    was_running = _server_process is not None

    if _server_process:
        try:
            _server_process.terminate()
            _server_process.wait(timeout=5)
        except Exception:
            try:
                _server_process.kill()
                _server_process.wait(timeout=3)
            except Exception:
                pass
        _server_process = None
        _current_model = None

    _kill_llama_on_port()

    if was_running:
        # Give the GPU driver a moment to reclaim VRAM after process death
        time.sleep(0.5)


# Register llama.cpp shutdown as a pre-model-load hook so that loading
# any AI model (TTS, ASR, Foley) will stop the server first to free VRAM.
try:
    from modules.core_components.ai_models.model_utils import register_pre_load_hook
    register_pre_load_hook(_stop_server)
except Exception:
    pass


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
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
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
    with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
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
        category="utility"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create the Prompt Manager UI."""
        from gradio_filelister import FileLister

        user_config = shared_state.get('_user_config', {})
        model_choices = _get_all_model_choices(user_config)

        components = {}

        with gr.TabItem("Prompt Manager"):
            with gr.Row():
                # Left column: prompt editor and saved prompts
                with gr.Column(scale=2):
                    gr.Markdown("### Prompt Editor")

                    components['prompt_text'] = gr.Textbox(
                        label="Prompt",
                        lines=8,
                        placeholder="Write your prompt here, or select a saved one from the list...",
                        interactive=True
                    )

                    with gr.Row():
                        components['save_btn'] = gr.Button(
                            "Save Prompt",
                            variant="primary",
                            scale=1
                        )
                        components['delete_btn'] = gr.Button(
                            "Delete",
                            variant="stop",
                            scale=1
                        )
                        components['clear_btn'] = gr.Button(
                            "Clear",
                            scale=1
                        )

                    components['pm_status'] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                    )

                    gr.Markdown("### Saved Prompts")
                    components['prompt_lister'] = FileLister(
                        value=_get_prompt_names(),
                        height=250,
                        show_footer=False,
                        interactive=True
                    )

                # Right column: LLM prompt generation
                with gr.Column(scale=2):
                    gr.Markdown("### LLM Prompt Generator")

                    components['llm_instruction'] = gr.Textbox(
                        label="Instructions for LLM",
                        lines=4,
                        placeholder="Describe what kind of prompt you want the LLM to generate...\ne.g., 'A dramatic monologue about a pirate finding treasure'",
                        interactive=True
                    )

                    components['system_prompt_preset'] = gr.Dropdown(
                        label="System Prompt Preset",
                        choices=SYSTEM_PROMPT_CHOICES,
                        value=SYSTEM_PROMPT_CHOICES[0],
                        interactive=True
                    )

                    components['system_prompt'] = gr.Textbox(
                        label="System Prompt",
                        lines=4,
                        value=SYSTEM_PROMPTS[SYSTEM_PROMPT_CHOICES[0]],
                        interactive=True
                    )

                    with gr.Row():
                        components['llm_model'] = gr.Dropdown(
                            label="LLM Model",
                            choices=model_choices,
                            value=model_choices[0] if model_choices else None,
                            interactive=True
                        )

                    with gr.Row():
                        components['generate_btn'] = gr.Button(
                            "Generate Prompt",
                            variant="primary"
                        )
                        components['stop_server_btn'] = gr.Button(
                            "Stop LLM Server",
                            variant="stop"
                        )

                    components['llm_status'] = gr.Textbox(
                        label="LLM Status",
                        interactive=False,
                        lines=1,
                    )

            # Hidden state
            components['pm_existing_files_json'] = gr.Textbox(visible=False)
            components['pm_suggested_name'] = gr.Textbox(visible=False)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Prompt Manager events."""
        user_config = shared_state.get('_user_config', {})
        show_input_modal_js = shared_state.get('show_input_modal_js')
        show_confirmation_modal_js = shared_state.get('show_confirmation_modal_js')
        input_trigger = shared_state.get('input_trigger')
        confirm_trigger = shared_state.get('confirm_trigger')

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

        components['prompt_lister'].change(
            load_selected_prompt,
            inputs=[components['prompt_lister']],
            outputs=[components['prompt_text']]
        )

        # --- Clear button ---
        components['clear_btn'].click(
            fn=lambda: (gr.update(value=""), ""),
            outputs=[components['prompt_text'], components['pm_status']]
        )

        # --- Save button: open input modal ---
        save_modal_js = show_input_modal_js(
            title="Save Prompt",
            message="Enter a name for this prompt:",
            placeholder="e.g., Dramatic Pirate, Thunder Storm, Calm Narrator",
            context="save_prompt_"
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
        components['save_btn'].click(
            fn=lambda lister_val: (get_existing_prompt_names(), get_selected_name(lister_val)),
            inputs=[components['prompt_lister']],
            outputs=[components['pm_existing_files_json'], components['pm_suggested_name']],
        ).then(
            fn=None,
            inputs=[components['pm_existing_files_json'], components['pm_suggested_name']],
            js=save_js
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
            raw_name = input_value[len("save_prompt_"):]
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
            inputs=[input_trigger, components['prompt_text']],
            outputs=[components['pm_status'], components['prompt_lister']]
        )

        # --- Delete button: confirm and delete ---
        components['delete_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Prompt?",
                message="This will permanently delete the selected prompt.",
                confirm_button_text="Delete",
                context="delete_prompt_"
            )
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
                return f"Deleted: {prompt_name}", gr.update(value=new_names), gr.update(value="")
            except Exception as e:
                return f"Error deleting: {e}", gr.update(), gr.update()

        confirm_trigger.change(
            handle_delete_prompt,
            inputs=[confirm_trigger, components['prompt_lister']],
            outputs=[components['pm_status'], components['prompt_lister'], components['prompt_text']]
        )

        # --- System prompt preset selector ---
        def on_preset_change(preset_name):
            if preset_name in SYSTEM_PROMPTS:
                return gr.update(value=SYSTEM_PROMPTS[preset_name])
            return gr.update()  # "Custom" — leave text as-is

        components['system_prompt_preset'].change(
            on_preset_change,
            inputs=[components['system_prompt_preset']],
            outputs=[components['system_prompt']]
        )

        # --- Refresh model list when dropdown is clicked ---
        def refresh_llm_models():
            choices = _get_all_model_choices(user_config)
            return gr.update(choices=choices)

        # --- Generate prompt with LLM ---
        def generate_with_llm(instruction, system_prompt, model_name, progress=gr.Progress()):
            """Send instruction to LLM and return generated prompt."""
            if not instruction or not instruction.strip():
                return gr.update(), "Please enter instructions for the LLM"

            if not model_name:
                return gr.update(), "Please select a model"

            try:
                progress(0.1, desc="Starting LLM server...")
                success, error = _start_server(model_name, user_config, progress)
                if not success:
                    return gr.update(), f"Server error: {error}"

                progress(0.6, desc="Generating prompt...")

                # Build request
                effective_system = system_prompt.strip() if system_prompt and system_prompt.strip() else SYSTEM_PROMPTS[SYSTEM_PROMPT_CHOICES[0]]
                payload = {
                    "messages": [
                        {"role": "system", "content": effective_system},
                        {"role": "user", "content": instruction.strip()}
                    ],
                    "stream": False,
                    "temperature": 0.8,
                    "top_p": 0.95,
                }

                response = requests.post(
                    f"http://localhost:{SERVER_PORT}/v1/chat/completions",
                    json=payload,
                    timeout=120
                )

                if response.status_code == 500:
                    # Server error, try restarting once
                    _stop_server()
                    success, error = _start_server(model_name, user_config, progress)
                    if not success:
                        return gr.update(), f"Server restart failed: {error}"
                    response = requests.post(
                        f"http://localhost:{SERVER_PORT}/v1/chat/completions",
                        json=payload,
                        timeout=120
                    )

                response.raise_for_status()
                data = response.json()

                # Extract response text
                choices = data.get("choices", [])
                if not choices:
                    return gr.update(), "LLM returned no response"

                generated_text = choices[0].get("message", {}).get("content", "").strip()
                if not generated_text:
                    return gr.update(), "LLM returned empty response"

                # Strip thinking tags if present
                if "<think>" in generated_text:
                    import re
                    generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

                progress(1.0, desc="Done")
                return gr.update(value=generated_text), "Prompt generated successfully"

            except requests.exceptions.ConnectionError:
                return gr.update(), "Could not connect to LLM server. Is llama-server installed?"
            except requests.exceptions.Timeout:
                return gr.update(), "LLM request timed out (120s)"
            except Exception as e:
                return gr.update(), f"Error: {e}"

        components['generate_btn'].click(
            generate_with_llm,
            inputs=[
                components['llm_instruction'],
                components['system_prompt'],
                components['llm_model'],
            ],
            outputs=[
                components['prompt_text'],
                components['llm_status'],
            ]
        )

        # --- Stop server button ---
        def stop_llm_server():
            _stop_server()
            return "LLM server stopped"

        components['stop_server_btn'].click(
            stop_llm_server,
            outputs=[components['llm_status']]
        )


# Export for registry
get_tool_class = lambda: PromptManagerTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(PromptManagerTool, port=7875, title="Prompt Manager - Standalone")
