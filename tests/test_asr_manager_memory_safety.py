import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from modules.core_components.ai_models import asr_manager as asr_module


class _FakeGenerateOutput:
    def __init__(self, sequences: torch.Tensor):
        self.sequences = sequences
        self.past_key_values = object()
        self.logits = object()
        self.hidden_states = object()
        self.attentions = object()
        self.scores = object()


class _FakeProcessor:
    def __call__(self, text, audio, return_tensors="pt", padding=True):
        batch = len(text)
        # input_ids should remain integer dtype after move/cast.
        return {
            "input_ids": torch.ones((batch, 3), dtype=torch.long),
            "input_features": torch.ones((batch, 4), dtype=torch.float32),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return [f"decoded_{int(row.sum().item())}" for row in token_ids]


class _FakeCoreModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.last_kwargs = None
        self.last_output = None
        self._cache = object()
        self.rope_deltas = object()
        self.thinker = SimpleNamespace(_cache=object(), rope_deltas=object())

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        batch = kwargs["input_ids"].shape[0]
        prompt = kwargs["input_ids"]
        completion = torch.tensor([[7, 8]], dtype=torch.long).repeat(batch, 1)
        out = _FakeGenerateOutput(torch.cat([prompt, completion], dim=1))
        self.last_output = out
        return out


class _FakeQwenRuntimeModel:
    def __init__(self):
        self.processor = _FakeProcessor()
        self.model = _FakeCoreModel()
        self.max_inference_batch_size = 8
        self.max_new_tokens = 512

    def _build_text_prompt(self, context: str, force_language=None):
        return context or "ctx"

    def _infer_asr_transformers(self, contexts, wavs, languages):
        raise RuntimeError("This method should be replaced by the low-memory patch")


def test_qwen_low_memory_patch_decodes_and_drops_large_generate_fields():
    model = _FakeQwenRuntimeModel()
    asr_module._patch_qwen3_asr_low_memory_runtime(model, aggressive_cleanup=False, debug_memory=False)

    outputs = model._infer_asr_transformers(
        contexts=["a", "b"],
        wavs=[np.zeros(320, dtype=np.float32), np.zeros(320, dtype=np.float32)],
        languages=[None, None],
    )

    assert outputs == ["decoded_15", "decoded_15"]
    # Ensure integer tensors were not cast to float dtype.
    assert model.model.last_kwargs["input_ids"].dtype == torch.long
    assert model.model.last_kwargs["input_features"].dtype == torch.float32
    # Ensure heavy generation outputs were explicitly nulled.
    assert model.model.last_output.past_key_values is None
    assert model.model.last_output.logits is None
    assert model.model.last_output.hidden_states is None
    assert model.model.last_output.attentions is None
    assert model.model.last_output.scores is None


class _FakeTranscribeModel:
    def __init__(self):
        self.max_inference_batch_size = 8
        self.call_batches = []
        self.calls = 0
        self.model = _FakeCoreModel()

    def transcribe(self, audio, language=None):
        self.calls += 1
        self.call_batches.append(self.max_inference_batch_size)
        if self.calls == 1:
            raise RuntimeError("CUDA out of memory")
        return [SimpleNamespace(text="ok", language="English")]


def test_qwen_wrapper_retries_once_with_halved_batch_on_oom(monkeypatch):
    fake = _FakeTranscribeModel()
    wrapper = asr_module._Qwen3ASRWrapper(
        fake,
        max_inference_batch_size=8,
        aggressive_cleanup=False,
        oom_retry=True,
        debug_memory=False,
    )

    monkeypatch.setattr(asr_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(asr_module, "empty_device_cache", lambda: None)

    result = wrapper.transcribe("dummy.wav", language="English")

    assert result == {"text": "ok", "language": "English"}
    assert fake.calls == 2
    assert fake.call_batches == [8, 4]
    # Wrapper restores configured batch size after retry.
    assert fake.max_inference_batch_size == 8


def test_qwen_wrapper_aggressive_cleanup_clears_generation_state(monkeypatch):
    class _SuccessModel:
        def __init__(self):
            self.max_inference_batch_size = 8
            self.model = _FakeCoreModel()

        def transcribe(self, audio, language=None):
            return [SimpleNamespace(text="done", language="English")]

    fake = _SuccessModel()
    wrapper = asr_module._Qwen3ASRWrapper(
        fake,
        max_inference_batch_size=8,
        aggressive_cleanup=True,
        oom_retry=False,
        debug_memory=False,
    )

    calls = {"cache": 0}

    def _cache():
        calls["cache"] += 1

    monkeypatch.setattr(asr_module, "empty_device_cache", _cache)

    result = wrapper.transcribe("dummy.wav")

    assert result == {"text": "done", "language": "English"}
    assert calls["cache"] >= 1
    assert fake.model._cache is None
    assert fake.model.rope_deltas is None
    assert fake.model.thinker._cache is None
    assert fake.model.thinker.rope_deltas is None


def test_library_manager_auto_split_has_finally_unload_forced_aligner():
    path = Path("modules/core_components/tools/library_manager.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    auto_split = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_auto_split_to_dataset":
            auto_split = node
            break

    assert auto_split is not None

    found_finally_unload = False
    for node in ast.walk(auto_split):
        if not isinstance(node, ast.Try):
            continue
        for fin in node.finalbody:
            if isinstance(fin, ast.Expr) and isinstance(fin.value, ast.Call):
                func = fin.value.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "unload_forced_aligner"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "asr_manager"
                ):
                    found_finally_unload = True
                    break
        if found_finally_unload:
            break

    assert found_finally_unload
