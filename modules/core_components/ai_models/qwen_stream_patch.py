"""Runtime patch for Qwen3-TTS native streaming voice-clone methods.

This module vendors the relevant API surface from:
https://github.com/dffdeeq/Qwen3-TTS-streaming

Patch behavior:
- No-op if methods already exist.
- Adds guarded wrappers to `Qwen3TTSModel`:
  - enable_streaming_optimizations(...)
  - stream_generate_pcm(...)
  - stream_generate_voice_clone(...)
- Adds low-level native stream entrypoints to
  `Qwen3TTSForConditionalGeneration` when missing:
  - _build_talker_inputs(...)
  - stream_generate_pcm(...)
- If underlying runtime still cannot stream, wrappers raise
  `NotImplementedError` so callers can safely fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class QwenStreamingPatchState:
    """Result of attempting to apply the Qwen streaming patch."""

    available: bool
    patched: bool
    reason: str = ""


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        topk = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        min_keep = topk.values[..., -1, None]
        logits = torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        inv_idx = torch.argsort(sorted_idx, dim=-1)
        logits = torch.gather(sorted_logits, dim=-1, index=inv_idx)
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    suppress_tokens: Optional[list[int]] = None,
) -> torch.Tensor:
    if suppress_tokens:
        logits = logits.clone()
        logits[..., suppress_tokens] = float("-inf")

    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _crossfade(prev_tail: np.ndarray, new_head: np.ndarray) -> np.ndarray:
    n = min(len(prev_tail), len(new_head))
    if n <= 0:
        return new_head
    w = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return prev_tail[:n] * (1.0 - w) + new_head[:n] * w


def _add_ref_code_context(
    window_codes: torch.Tensor,
    ref_code_context: Optional[torch.Tensor],
    ref_code_frames: int,
    decode_window_frames: int,
) -> tuple[torch.Tensor, int]:
    if ref_code_context is None or window_codes.shape[0] >= decode_window_frames:
        return window_codes, 0

    available_space = decode_window_frames - window_codes.shape[0]
    ref_prefix_frames = min(available_space, ref_code_frames)
    if ref_prefix_frames > 0:
        ref_prefix = ref_code_context[-ref_prefix_frames:]
        return torch.cat([ref_prefix, window_codes], dim=0), ref_prefix_frames

    return window_codes, 0


def _maybe_mark_cudagraph_step_begin() -> None:
    marker = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if callable(marker):
        try:
            marker()
        except Exception:
            pass


def _model_supports_native_streaming(instance: Any) -> bool:
    model = getattr(instance, "model", None)
    if model is None:
        return False
    return callable(getattr(model, "stream_generate_pcm", None)) or callable(getattr(model, "stream_generate", None))


def qwen_streaming_runtime_available(instance: Any) -> bool:
    """Return whether a Qwen model instance can perform native streaming."""
    if not hasattr(instance, "stream_generate_voice_clone"):
        return False

    model = getattr(instance, "model", None)
    if model is None:
        return False

    if callable(getattr(model, "stream_generate_pcm", None)):
        talker = getattr(model, "talker", None)
        speech_tokenizer = getattr(model, "speech_tokenizer", None)
        return bool(
            talker is not None
            and callable(getattr(talker, "forward", None))
            and speech_tokenizer is not None
            and callable(getattr(speech_tokenizer, "decode", None))
        )

    return callable(getattr(model, "stream_generate", None))


def _patch_qwen_core_streaming() -> tuple[bool, str]:
    """Patch low-level Qwen core model with stream methods when missing."""
    try:
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    except Exception as exc:
        return False, f"qwen core unavailable: {str(exc)}"

    patched = False

    if not hasattr(Qwen3TTSForConditionalGeneration, "enable_streaming_optimizations"):

        def enable_streaming_optimizations(
            self,
            decode_window_frames: int = 80,
            use_compile: bool = True,
            use_cuda_graphs: bool = True,
            compile_mode: str = "reduce-overhead",
            use_fast_codebook: bool = False,
            compile_codebook_predictor: bool = True,
            compile_talker: bool = True,
        ):
            del decode_window_frames, use_compile, use_cuda_graphs, compile_mode
            del use_fast_codebook, compile_codebook_predictor, compile_talker

            speech_tokenizer = getattr(self, "speech_tokenizer", None)
            if speech_tokenizer is not None:
                fn = getattr(speech_tokenizer, "enable_streaming_optimizations", None)
                if callable(fn):
                    try:
                        fn(
                            decode_window_frames=decode_window_frames,
                            use_compile=use_compile,
                            use_cuda_graphs=use_cuda_graphs,
                            compile_mode=compile_mode,
                        )
                    except Exception:
                        pass

            talker = getattr(self, "talker", None)
            if talker is not None:
                if use_fast_codebook:
                    fn = getattr(talker, "enable_fast_codebook_gen", None)
                    if callable(fn):
                        try:
                            fn(True)
                        except Exception:
                            pass
                if compile_talker and use_compile:
                    fn = getattr(talker, "enable_compile", None)
                    if callable(fn):
                        try:
                            fn(mode="default")
                        except Exception:
                            pass
                if compile_codebook_predictor and use_compile:
                    code_predictor = getattr(talker, "code_predictor", None)
                    fn = getattr(code_predictor, "enable_compile", None)
                    if callable(fn):
                        try:
                            fn(mode=compile_mode)
                        except Exception:
                            pass
            return self

        Qwen3TTSForConditionalGeneration.enable_streaming_optimizations = enable_streaming_optimizations
        patched = True

    if not hasattr(Qwen3TTSForConditionalGeneration, "_build_talker_inputs"):

        def _build_talker_inputs(
            self,
            input_ids: list[torch.Tensor],
            instruct_ids: Optional[list[torch.Tensor]],
            ref_ids: Optional[list[torch.Tensor]],
            voice_clone_prompt: Optional[list[dict]],
            languages: list[str],
            speakers: Optional[list[str]],
            non_streaming_mode: bool = False,
        ):
            talker_input_embeds = [[] for _ in range(len(input_ids))]

            voice_clone_spk_embeds = None
            if voice_clone_prompt is not None:
                voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

            if instruct_ids is not None:
                for index, instruct_id in enumerate(instruct_ids):
                    if instruct_id is not None:
                        talker_input_embeds[index].append(
                            self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
                        )

            trailing_text_hiddens = []
            if speakers is None:
                speakers = [None] * len(input_ids)

            for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
                if voice_clone_spk_embeds is None:
                    if speaker == "" or speaker is None:
                        speaker_embed = None
                    else:
                        if speaker.lower() not in self.config.talker_config.spk_id:
                            raise NotImplementedError(f"Speaker {speaker} not implemented")
                        spk_id = self.config.talker_config.spk_id[speaker.lower()]
                        speaker_embed = self.talker.get_input_embeddings()(
                            torch.tensor(spk_id, device=self.talker.device, dtype=input_id.dtype)
                        )
                else:
                    if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                        speaker_embed = voice_clone_spk_embeds[index]
                    else:
                        speaker_embed = None

                if language.lower() == "auto":
                    language_id = None
                else:
                    if language.lower() not in self.config.talker_config.codec_language_id:
                        raise NotImplementedError(f"Language {language} not implemented")
                    language_id = self.config.talker_config.codec_language_id[language.lower()]

                if (
                    language.lower() in ["chinese", "auto"]
                    and speaker != ""
                    and speaker is not None
                    and self.config.talker_config.spk_is_dialect[speaker.lower()] is not False
                ):
                    dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                    language_id = self.config.talker_config.codec_language_id[dialect]

                tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                    self.talker.get_text_embeddings()(
                        torch.tensor(
                            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
                ).chunk(3, dim=1)

                if language_id is None:
                    codec_prefill_list = [[
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]]
                else:
                    codec_prefill_list = [[
                        self.config.talker_config.codec_think_id,
                        self.config.talker_config.codec_think_bos_id,
                        language_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]]

                codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                    torch.tensor(codec_prefill_list, device=self.talker.device, dtype=input_id.dtype)
                )
                codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                    torch.tensor(
                        [[self.config.talker_config.codec_pad_id, self.config.talker_config.codec_bos_id]],
                        device=self.talker.device,
                        dtype=input_id.dtype,
                    )
                )

                if speaker_embed is None:
                    codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
                else:
                    codec_input_emebdding = torch.cat(
                        [codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1],
                        dim=1,
                    )

                _talker_input_embed_role = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, :3]))

                _talker_input_embed = torch.cat(
                    (tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed),
                    dim=1,
                ) + codec_input_emebdding[:, :-1]

                talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

                if (
                    voice_clone_prompt is not None
                    and voice_clone_prompt["ref_code"] is not None
                    and voice_clone_prompt["icl_mode"][index]
                ):
                    icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                        text_id=input_id[:, 3:-5],
                        ref_id=ref_ids[index][:, 3:-2],
                        ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                        tts_pad_embed=tts_pad_embed,
                        tts_eos_embed=tts_eos_embed,
                        non_streaming_mode=non_streaming_mode,
                    )
                    talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
                else:
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4]))
                            + codec_input_emebdding[:, -1:],
                        ],
                        dim=1,
                    )
                    if non_streaming_mode:
                        talker_input_embed = talker_input_embed[:, :-1]
                        talker_input_embed = torch.cat(
                            [
                                talker_input_embed,
                                torch.cat(
                                    (
                                        self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:-5])),
                                        tts_eos_embed,
                                    ),
                                    dim=1,
                                )
                                + self.talker.get_input_embeddings()(
                                    torch.tensor(
                                        [[self.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                                        device=self.talker.device,
                                        dtype=input_id.dtype,
                                    )
                                ),
                                tts_pad_embed
                                + self.talker.get_input_embeddings()(
                                    torch.tensor(
                                        [[self.config.talker_config.codec_bos_id]],
                                        device=self.talker.device,
                                        dtype=input_id.dtype,
                                    )
                                ),
                            ],
                            dim=1,
                        )
                        trailing_text_hidden = tts_pad_embed
                    else:
                        trailing_text_hidden = torch.cat(
                            (self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 4:-5])), tts_eos_embed),
                            dim=1,
                        )

                talker_input_embeds[index].append(talker_input_embed)
                trailing_text_hiddens.append(trailing_text_hidden)

            for index, talker_input_embed in enumerate(talker_input_embeds):
                talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

            original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
            sequences = [t.squeeze(0) for t in talker_input_embeds]
            sequences_reversed = [t.flip(dims=[0]) for t in sequences]
            padded_reversed = torch.nn.utils.rnn.pad_sequence(sequences_reversed, batch_first=True, padding_value=0.0)
            talker_input_embeds = padded_reversed.flip(dims=[1])

            batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
            indices = torch.arange(max_len).expand(batch_size, -1)
            num_pads = max_len - original_lengths
            talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

            pad_embedding_vector = tts_pad_embed.squeeze()
            sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
            trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
            padded_hiddens = torch.nn.utils.rnn.pad_sequence(sequences_to_pad, batch_first=True, padding_value=0.0)
            arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
                len(trailing_text_original_lengths), -1
            )
            lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
            padding_mask = arange_tensor >= lengths_tensor
            padded_hiddens[padding_mask] = pad_embedding_vector
            trailing_text_hiddens = padded_hiddens

            return talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed

        Qwen3TTSForConditionalGeneration._build_talker_inputs = _build_talker_inputs
        patched = True

    if not hasattr(Qwen3TTSForConditionalGeneration, "stream_generate_pcm"):

        @torch.inference_mode()
        def stream_generate_pcm(
            self,
            input_ids: list[torch.Tensor],
            instruct_ids: Optional[list[torch.Tensor]] = None,
            ref_ids: Optional[list[torch.Tensor]] = None,
            voice_clone_prompt: Optional[list[dict]] = None,
            languages: Optional[list[str]] = None,
            speakers: Optional[list[str]] = None,
            non_streaming_mode: bool = False,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 1.0,
            temperature: float = 0.9,
            subtalker_dosample: bool = True,
            subtalker_top_k: int = 50,
            subtalker_top_p: float = 1.0,
            subtalker_temperature: float = 0.9,
            emit_every_frames: int = 8,
            decode_window_frames: int = 80,
            overlap_samples: int = 0,
            max_frames: int = 10000,
            use_optimized_decode: bool = True,
        ) -> Iterator[tuple[np.ndarray, int]]:
            if languages is None:
                languages = ["Auto"] * len(input_ids)

            talker_input_embeds, talker_attention_mask, trailing_text_hiddens, tts_pad_embed = self._build_talker_inputs(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt,
                languages=languages,
                speakers=speakers,
                non_streaming_mode=non_streaming_mode,
            )

            eos_id = self.config.talker_config.codec_eos_token_id
            vocab_size = self.config.talker_config.vocab_size
            suppress_tokens = [i for i in range(vocab_size - 1024, vocab_size) if i != eos_id]

            _maybe_mark_cudagraph_step_begin()

            out = self.talker.forward(
                inputs_embeds=talker_input_embeds,
                attention_mask=talker_attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                trailing_text_hidden=trailing_text_hiddens,
                tts_pad_embed=tts_pad_embed,
                generation_step=None,
                past_hidden=None,
                past_key_values=None,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
            )

            past_key_values = out.past_key_values
            past_hidden = out.past_hidden
            generation_step = out.generation_step

            last_logits = out.logits[:, -1, :]
            if do_sample:
                token = _sample_next_token(last_logits, temperature, top_k, top_p, suppress_tokens)
            else:
                token = torch.argmax(last_logits, dim=-1)

            ref_code_context = None
            ref_code_frames = 0
            if voice_clone_prompt is not None:
                ref_code_list = voice_clone_prompt.get("ref_code", None)
                icl_mode_list = voice_clone_prompt.get("icl_mode", None)
                if ref_code_list is not None and icl_mode_list is not None:
                    if ref_code_list[0] is not None and icl_mode_list[0]:
                        ref_code_context = ref_code_list[0].to(self.talker.device)
                        ref_code_frames = ref_code_context.shape[0]

            codes_buffer: list[torch.Tensor] = []
            decoded_tail: Optional[np.ndarray] = None
            frames_since_emit = 0
            total_frames_emitted = 0

            for _ in range(int(max_frames)):
                _maybe_mark_cudagraph_step_begin()

                step_out = self.talker.forward(
                    input_ids=token.unsqueeze(1),
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=False,
                    past_key_values=past_key_values,
                    past_hidden=past_hidden,
                    generation_step=generation_step,
                    trailing_text_hidden=trailing_text_hiddens,
                    tts_pad_embed=tts_pad_embed,
                    subtalker_dosample=subtalker_dosample,
                    subtalker_top_k=subtalker_top_k,
                    subtalker_top_p=subtalker_top_p,
                    subtalker_temperature=subtalker_temperature,
                )

                past_key_values = step_out.past_key_values
                past_hidden = step_out.past_hidden
                generation_step = step_out.generation_step

                codec_ids = step_out.hidden_states[1]
                if codec_ids is None:
                    break

                if codec_ids[0, 0] == eos_id:
                    break

                codes_buffer.append(codec_ids[0].detach())

                step_logits = step_out.logits[:, -1, :]
                if do_sample:
                    token = _sample_next_token(step_logits, temperature, top_k, top_p, suppress_tokens)
                else:
                    token = torch.argmax(step_logits, dim=-1)

                frames_since_emit += 1
                if frames_since_emit < int(max(1, emit_every_frames)):
                    continue
                frames_since_emit = 0

                start = max(0, len(codes_buffer) - int(max(1, decode_window_frames)))
                window_codes = torch.stack(codes_buffer[start:], dim=0)
                window, _ = _add_ref_code_context(
                    window_codes,
                    ref_code_context,
                    ref_code_frames,
                    int(max(1, decode_window_frames)),
                )

                if use_optimized_decode and hasattr(self.speech_tokenizer, "decode_streaming"):
                    try:
                        wavs, sr = self.speech_tokenizer.decode_streaming(
                            window.to(self.talker.device),
                            use_optimized=True,
                            pad_to_size=int(max(1, decode_window_frames)),
                        )
                    except Exception:
                        wavs, sr = self.speech_tokenizer.decode([{"audio_codes": window.to(self.talker.device)}])
                else:
                    wavs, sr = self.speech_tokenizer.decode([{"audio_codes": window.to(self.talker.device)}])

                wav = wavs[0].astype(np.float32)
                get_rate = getattr(self.speech_tokenizer, "get_decode_upsample_rate", None)
                if callable(get_rate):
                    samples_per_frame = int(get_rate())
                else:
                    samples_per_frame = max(1, int(len(wav) / max(int(window.shape[0]), 1)))

                step_samples = samples_per_frame * int(max(1, emit_every_frames))
                chunk = wav[-step_samples:] if step_samples > 0 else wav

                if decoded_tail is not None and overlap_samples > 0:
                    ov = min(int(overlap_samples), len(decoded_tail), len(chunk))
                    if ov > 0:
                        head = _crossfade(decoded_tail[-ov:], chunk[:ov])
                        chunk = np.concatenate([head, chunk[ov:]], axis=0)

                decoded_tail = chunk.copy()
                total_frames_emitted = len(codes_buffer)
                yield chunk, int(sr)

            remaining_frames = len(codes_buffer) - total_frames_emitted
            if remaining_frames > 0:
                context_frames = min(total_frames_emitted, int(max(1, decode_window_frames)) - remaining_frames)
                start_idx = total_frames_emitted - context_frames
                window_codes = torch.stack(codes_buffer[start_idx:], dim=0)

                window, flush_ref_prefix_frames = _add_ref_code_context(
                    window_codes,
                    ref_code_context,
                    ref_code_frames,
                    int(max(1, decode_window_frames)),
                )

                wavs, sr = self.speech_tokenizer.decode([{"audio_codes": window.to(self.talker.device)}])
                wav = wavs[0].astype(np.float32)

                skip_frames = flush_ref_prefix_frames + context_frames
                if skip_frames > 0:
                    samples_per_frame = len(wav) / max(int(window.shape[0]), 1)
                    skip_samples = int(skip_frames * samples_per_frame)
                    wav = wav[skip_samples:]

                if decoded_tail is not None and overlap_samples > 0 and len(wav) > 0:
                    ov = min(int(overlap_samples), len(decoded_tail), len(wav))
                    if ov > 0:
                        head = _crossfade(decoded_tail[-ov:], wav[:ov])
                        wav = np.concatenate([head, wav[ov:]], axis=0)

                if len(wav) > 0:
                    yield wav, int(sr)

        Qwen3TTSForConditionalGeneration.stream_generate_pcm = stream_generate_pcm
        patched = True

    return patched, ""


def apply_qwen_streaming_patch() -> QwenStreamingPatchState:
    """Apply guarded streaming wrappers to Qwen3TTSModel if needed."""
    try:
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except Exception as exc:
        return QwenStreamingPatchState(
            available=False,
            patched=False,
            reason=f"qwen_tts unavailable: {str(exc)}",
        )

    patched = False

    core_patched, core_reason = _patch_qwen_core_streaming()
    patched = patched or core_patched

    if not hasattr(Qwen3TTSModel, "enable_streaming_optimizations"):

        def enable_streaming_optimizations(self, **kwargs):
            model = getattr(self, "model", None)
            fn = getattr(model, "enable_streaming_optimizations", None)
            if callable(fn):
                try:
                    fn(**kwargs)
                except Exception:
                    pass
            setattr(self, "_vcs_streaming_optimizations", dict(kwargs))
            return self

        Qwen3TTSModel.enable_streaming_optimizations = enable_streaming_optimizations
        patched = True

    if not hasattr(Qwen3TTSModel, "stream_generate_pcm"):

        def stream_generate_pcm(
            self,
            inputs,
            output_format: str = "pcm",
            chunk_size: int = 16,
            speech_rate: float = 1.0,
            return_numpy: bool = True,
            seed: int | None = None,
            do_sample: bool = True,
            temperature: float = 0.7,
            top_k: int = 20,
            top_p: float = 0.8,
            repetition_penalty: float = 1.1,
            max_new_tokens: int = 2048,
            play_steps: int = 50,
        ) -> Iterator[tuple[Any, int]]:
            del speech_rate, return_numpy, seed, repetition_penalty, max_new_tokens

            if output_format != "pcm":
                raise NotImplementedError("stream_generate_pcm currently supports output_format='pcm' only.")

            model = getattr(self, "model", None)
            if model is None:
                raise NotImplementedError("Native Qwen streaming not available in current runtime.")

            # Legacy fallback path used by tests and older forks.
            legacy_stream = getattr(model, "stream_generate", None)
            if callable(legacy_stream):
                stream_iter = legacy_stream(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    speaker_embedding=inputs.get("speaker_embedding"),
                    text_tokens_in_input_ids=inputs.get("text_tokens_in_input_ids"),
                    output_format=output_format,
                    chunk_size=chunk_size,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    play_steps=play_steps,
                )
                for audio, sr in stream_iter:
                    yield audio, sr
                return

            raise NotImplementedError("Native Qwen streaming not available in current runtime.")

        Qwen3TTSModel.stream_generate_pcm = stream_generate_pcm
        patched = True

    if not hasattr(Qwen3TTSModel, "stream_generate_voice_clone"):

        @torch.inference_mode()
        def stream_generate_voice_clone(self, **kwargs) -> Iterator[tuple[Any, int]]:
            model = getattr(self, "model", None)
            if model is None:
                raise NotImplementedError("Native Qwen streaming not available in current runtime.")

            # Modern low-level stream path (fork-derived)
            if callable(getattr(model, "stream_generate_pcm", None)):
                text = kwargs.get("text")
                if isinstance(text, list):
                    raise ValueError("stream_generate_voice_clone only supports single text, not batch")
                if text is None:
                    raise ValueError("text is required")

                language = kwargs.get("language") if kwargs.get("language") is not None else "Auto"
                voice_clone_prompt = kwargs.get("voice_clone_prompt")

                if voice_clone_prompt is None:
                    ref_audio = kwargs.get("ref_audio")
                    if ref_audio is None:
                        raise ValueError("Either voice_clone_prompt or ref_audio must be provided")
                    prompt_items = self.create_voice_clone_prompt(
                        ref_audio=ref_audio,
                        ref_text=kwargs.get("ref_text"),
                        x_vector_only_mode=kwargs.get("x_vector_only_mode", False),
                    )
                    voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                    ref_texts_for_ids = [prompt_items[0].ref_text]
                elif isinstance(voice_clone_prompt, list):
                    prompt_items = voice_clone_prompt
                    if (
                        len(prompt_items) > 0
                        and not isinstance(prompt_items[0], dict)
                        and hasattr(prompt_items[0], "ref_text")
                        and hasattr(self, "_prompt_items_to_voice_clone_prompt")
                    ):
                        voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                        ref_texts_for_ids = [prompt_items[0].ref_text]
                    elif len(prompt_items) > 0 and isinstance(prompt_items[0], dict):
                        voice_clone_prompt_dict = prompt_items[0]
                        ref_texts_for_ids = [None]
                    else:
                        voice_clone_prompt_dict = voice_clone_prompt
                        ref_texts_for_ids = [None]
                elif isinstance(voice_clone_prompt, dict):
                    voice_clone_prompt_dict = voice_clone_prompt
                    ref_texts_for_ids = None
                elif hasattr(voice_clone_prompt, "ref_text") and hasattr(self, "_prompt_items_to_voice_clone_prompt"):
                    voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt([voice_clone_prompt])
                    ref_texts_for_ids = [voice_clone_prompt.ref_text]
                else:
                    raise ValueError("Unsupported voice_clone_prompt type for streaming")

                texts = [text]
                languages = [language]

                validate_languages = getattr(self, "_validate_languages", None)
                if callable(validate_languages):
                    validate_languages(languages)

                if not (hasattr(self, "_build_assistant_text") and hasattr(self, "_tokenize_texts")):
                    raise NotImplementedError("Qwen streaming patch missing tokenizer helpers on runtime wrapper.")

                input_texts = [self._build_assistant_text(t) for t in texts]
                input_ids = self._tokenize_texts(input_texts)

                ref_ids = None
                if ref_texts_for_ids is not None:
                    ref_ids = []
                    for rt in ref_texts_for_ids:
                        if rt is None or rt == "":
                            ref_ids.append(None)
                        else:
                            if not hasattr(self, "_build_ref_text"):
                                ref_ids.append(None)
                            else:
                                ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                                ref_ids.append(ref_tok)

                merge_generate_kwargs = getattr(self, "_merge_generate_kwargs", None)
                if callable(merge_generate_kwargs):
                    gen_kwargs = merge_generate_kwargs(**kwargs)
                else:
                    gen_kwargs = dict(kwargs)

                supported_params = {
                    "do_sample",
                    "top_k",
                    "top_p",
                    "temperature",
                    "subtalker_dosample",
                    "subtalker_top_k",
                    "subtalker_top_p",
                    "subtalker_temperature",
                }
                gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in supported_params}

                emit_every_frames = int(kwargs.get("chunk_size", kwargs.get("emit_every_frames", 8)))
                decode_window_frames = int(kwargs.get("play_steps", kwargs.get("decode_window_frames", 80)))

                for chunk, sr in model.stream_generate_pcm(
                    input_ids=input_ids,
                    ref_ids=ref_ids,
                    voice_clone_prompt=voice_clone_prompt_dict,
                    languages=languages,
                    non_streaming_mode=bool(kwargs.get("non_streaming_mode", False)),
                    emit_every_frames=max(1, emit_every_frames),
                    decode_window_frames=max(1, decode_window_frames),
                    overlap_samples=int(kwargs.get("overlap_samples", 0)),
                    max_frames=int(kwargs.get("max_frames", 10000)),
                    use_optimized_decode=bool(kwargs.get("use_optimized_decode", True)),
                    **gen_kwargs,
                ):
                    yield chunk, sr
                return

            # Legacy fallback path used by older runtimes/tests.
            if not _model_supports_native_streaming(self):
                raise NotImplementedError("Native Qwen streaming not available in current runtime.")

            voice_clone_prompt = kwargs.get("voice_clone_prompt")
            if voice_clone_prompt is None:
                ref_audio = kwargs.get("ref_audio")
                ref_text = kwargs.get("ref_text")
                if not ref_audio or not ref_text:
                    raise ValueError("Need either voice_clone_prompt or both ref_audio and ref_text.")
                voice_clone_prompt = self.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=kwargs.get("x_vector_only_mode", False),
                )
                kwargs["voice_clone_prompt"] = voice_clone_prompt

            if not hasattr(self, "_convert_inputs_2_neural_method_inputs"):
                raise NotImplementedError("Qwen runtime missing compatible streaming conversion method.")

            neural_inputs = self._convert_inputs_2_neural_method_inputs(kwargs, model_type="voice_clone")
            for audio, sr in self.stream_generate_pcm(
                inputs=neural_inputs,
                output_format=kwargs.get("output_format", "pcm"),
                chunk_size=int(kwargs.get("chunk_size", 16)),
                speech_rate=float(kwargs.get("speech_rate", 1.0)),
                return_numpy=bool(kwargs.get("return_numpy", True)),
                seed=kwargs.get("seed"),
                do_sample=bool(kwargs.get("do_sample", True)),
                temperature=float(kwargs.get("temperature", 0.7)),
                top_k=int(kwargs.get("top_k", 20)),
                top_p=float(kwargs.get("top_p", 0.8)),
                repetition_penalty=float(kwargs.get("repetition_penalty", 1.1)),
                max_new_tokens=int(kwargs.get("max_new_tokens", 2048)),
                play_steps=int(kwargs.get("play_steps", 50)),
            ):
                yield audio, sr

        Qwen3TTSModel.stream_generate_voice_clone = stream_generate_voice_clone
        patched = True

    reason = core_reason
    return QwenStreamingPatchState(available=True, patched=patched, reason=reason)
