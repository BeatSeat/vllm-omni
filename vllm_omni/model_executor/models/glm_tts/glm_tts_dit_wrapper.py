# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS DiT wrapper: LLM_GENERATION-compatible flow-matching + vocoder."""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.glm_tts.glm_tts import (
    resolve_glm_tts_model_dir,
)
from vllm_omni.model_executor.models.glm_tts.vocoder import (
    load_vocoder,
    mel_to_audio,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


_GLM_TTS_OFFICIAL_FLOW_FALLBACK: dict[str, Any] = {
    "spkr_emb_adaLN": True,
    "speech_token_cfg": False,
    "remove_spkr_concat_condition": True,
    "mel_dim": 80,
    "mel_framerate": 50,
    "input_frame_rate": 25,
}

_GLM_TTS_RUNTIME_FLOW_DEFAULTS: dict[str, Any] = {
    "sample_rate": 24000,
    "n_timesteps": 10,
    "t_scheduler": "cosine",
    "inference_cfg_rate": 0.7,
    "spk_embed_dim": 80,
    "trans_dim": 768,
    "depth": 18,
    "heads": 12,
}


def _extract_flow_config_dict(config: object) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    flow_config = config.get("flow", config)
    return dict(flow_config) if isinstance(flow_config, dict) else {}


def _load_tag_stripped_yaml_config(path: str) -> dict[str, Any]:
    """Load GLM-TTS HyperPyYAML config without instantiating objects.

    The official GLM-TTS flow config is model-local HyperPyYAML, e.g.
    ``flow: !new:flow.flow.Flow``.  This is intentionally not routed through
    the project deployment YAML loader because we only need constructor kwargs,
    not Python object creation.  Several Omni model wrappers already use
    model-local loaders for non-standard sub-checkpoints; this mirrors that
    pattern for GLM-TTS flow metadata.
    """
    import yaml

    class GLMTTSFlowConfigLoader(yaml.SafeLoader):
        pass

    def construct_unknown_tag(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> object:
        del tag_suffix
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    GLMTTSFlowConfigLoader.add_multi_constructor("!", construct_unknown_tag)
    with open(path) as f:
        loaded = yaml.load(f, Loader=GLMTTSFlowConfigLoader)
    return loaded if isinstance(loaded, dict) else {}


def _load_glm_tts_flow_config(flow_dir: str) -> tuple[dict[str, Any], dict[str, str]]:
    config: dict[str, Any] = dict(_GLM_TTS_OFFICIAL_FLOW_FALLBACK)
    sources = {
        "fallback": "official GLM-TTS flow/config.yaml fields",
        "json": "missing",
        "yaml": "missing",
    }

    if os.path.exists(cj := os.path.join(flow_dir, "config.json")):
        try:
            with open(cj) as f:
                json_config = json.load(f)
            json_flow_config = _extract_flow_config_dict(json_config)
            if json_flow_config:
                config.update(json_flow_config)
                sources["json"] = cj
            else:
                sources["json"] = f"{cj} (no flow mapping)"
        except Exception as e:
            sources["json"] = f"{cj} (failed: {e})"
            logger.warning("Failed to parse GLM-TTS flow config.json: %s", e)

    if os.path.exists(cy := os.path.join(flow_dir, "config.yaml")):
        try:
            yaml_config = _load_tag_stripped_yaml_config(cy)
            yaml_flow_config = _extract_flow_config_dict(yaml_config)
            if yaml_flow_config:
                config.update(yaml_flow_config)
                sources["yaml"] = cy
            else:
                sources["yaml"] = f"{cy} (no flow mapping)"
        except Exception as e:
            sources["yaml"] = f"{cy} (failed: {e})"
            logger.warning(
                "Failed to parse GLM-TTS flow config.yaml; using official GLM-TTS fallback flow config: %s",
                e,
            )

    return config, sources


def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    seq_range = torch.arange(0, max_len, device=lengths.device)
    return seq_range.unsqueeze(0).expand(lengths.shape[0], max_len) >= lengths.unsqueeze(1)


def as_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, torch.Tensor) else None


def as_bool(value: object) -> bool:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, torch.Tensor) and value.numel() > 0:
        return bool(value.reshape(-1)[0].item())
    return bool(value) if value is not None else False


def as_str(value: object) -> str | None:
    if isinstance(value, list) and value:
        value = value[0]
    return str(value) if value is not None else None


def valid_speech_tokens(
    value: object,
    *,
    device: torch.device,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return GLM-TTS speech tokens, treating an explicit empty list as empty."""
    if value is None:
        token = fallback if fallback is not None else torch.empty(0, device=device, dtype=torch.long)
    elif isinstance(value, torch.Tensor):
        token = value.to(device=device, dtype=torch.long)
    elif isinstance(value, np.ndarray):
        token = torch.as_tensor(value, device=device, dtype=torch.long)
    elif isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], torch.Tensor):
            token = value[0].to(device=device, dtype=torch.long)
        elif value and all(isinstance(item, torch.Tensor) for item in value):
            token = torch.cat([item.reshape(-1).to(device=device, dtype=torch.long) for item in value])
        else:
            token = torch.tensor(value, device=device, dtype=torch.long)
    else:
        token = fallback if fallback is not None else torch.empty(0, device=device, dtype=torch.long)

    token = token.reshape(-1).to(device=device, dtype=torch.long)
    return token[(token >= 0) & (token < 32768)]


def split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
    if seq_token_counts is not None:
        boundaries, total = [0], ids.numel()
        for count in seq_token_counts:
            boundaries.append(boundaries[-1] + int(count))
        return [ids[boundaries[i] : min(boundaries[i + 1], total)] for i in range(len(seq_token_counts))]
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + int(s))
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
    except ImportError:
        pass
    return [ids]


class GLMTTSDiTForGeneration(nn.Module):
    """GLM-TTS DiT flow-matching stage wrapped as LLM_GENERATION."""

    have_multimodal_outputs = True
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model_stage = "glm_tts_dit"
        self.max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        self._use_dit_cuda_graphs = bool(getattr(config, "use_dit_cuda_graphs", False))
        self._dit_cudagraph: CUDAGraphGLMTTSDiTWrapper | None = None
        self._codec_chunk_frames, self._codec_left_context_frames = self._connector_chunk_config(vllm_config)

        model_path = vllm_config.model_config.model
        model_path = resolve_glm_tts_model_dir(
            model_path,
            tokenizer_path=getattr(vllm_config.model_config, "tokenizer", None),
            required_files=("flow/flow.pt",),
            optional_files=("hift/hift.pt", "vocos2d/generator_jit.ckpt"),
        )
        self.model_dir = model_path
        flow_dir = os.path.join(self.model_dir, "flow")
        self.hf_flow_config, flow_config_sources = _load_glm_tts_flow_config(flow_dir)
        logger.info("GLM-TTS flow config sources: %s", flow_config_sources)
        logger.info("GLM-TTS flow config effective: %s", self.hf_flow_config)

        self.mel_dim = self.hf_flow_config.get("mel_dim", 80)
        self.input_frame_rate = self.hf_flow_config.get("input_frame_rate", 25.0)
        self.mel_framerate = self.hf_flow_config.get("mel_framerate", 50)
        self.sample_rate = self.hf_flow_config.get("sample_rate", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["sample_rate"])

        self.n_timesteps = self.hf_flow_config.get("n_timesteps", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["n_timesteps"])
        self.t_scheduler = self.hf_flow_config.get("t_scheduler", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["t_scheduler"])
        self.inference_cfg_rate = self.hf_flow_config.get(
            "inference_cfg_rate", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["inference_cfg_rate"]
        )
        self.speech_token_cfg = self.hf_flow_config.get("speech_token_cfg", False)

        self.spk_embed_dim = self.hf_flow_config.get("spk_embed_dim", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["spk_embed_dim"])
        self.spkr_emb_adaln = self.hf_flow_config.get("spkr_emb_adaLN", True)
        self.remove_spkr_concat_condition = self.hf_flow_config.get("remove_spkr_concat_condition", True)

        logger.info(
            "GLM-TTS flow runtime: input_frame_rate=%s mel_framerate=%s mel_dim=%s sample_rate=%s "
            "n_timesteps=%s t_scheduler=%s inference_cfg_rate=%s speech_token_cfg=%s "
            "spkr_emb_adaLN=%s remove_spkr_concat_condition=%s",
            self.input_frame_rate,
            self.mel_framerate,
            self.mel_dim,
            self.sample_rate,
            self.n_timesteps,
            self.t_scheduler,
            self.inference_cfg_rate,
            self.speech_token_cfg,
            self.spkr_emb_adaln,
            self.remove_spkr_concat_condition,
        )

        from vllm_omni.model_executor.models.glm_tts.glm_tts_dit import GLMTTSDiT

        trans_dim = self.hf_flow_config.get("trans_dim", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["trans_dim"])
        depth = self.hf_flow_config.get("depth", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["depth"])
        heads = self.hf_flow_config.get("heads", _GLM_TTS_RUNTIME_FLOW_DEFAULTS["heads"])
        logger.info("GLM-TTS DiT architecture: trans_dim=%s depth=%s heads=%s", trans_dim, depth, heads)
        condition_dim = self.mel_dim if self.remove_spkr_concat_condition else self.mel_dim + self.spk_embed_dim

        self.dit = GLMTTSDiT(
            trans_dim=trans_dim,
            depth=depth,
            heads=heads,
            dim_head=64,
            ff_mult=2,
            dropout=0.1,
            mel_dim=self.mel_dim,
            text_vocab_size=100000,
            text_emb_dim=512,
            conv_layers=4,
            condition_dim=condition_dim,
            spkr_emb_adaln=self.spkr_emb_adaln,
            spkr_dim=192,
            use_wavlm_emb=False,
        )
        self._model_dtype = self._resolve_model_dtype(vllm_config)
        if self._model_dtype is not None:
            self.dit.to(dtype=self._model_dtype)

        if not self.remove_spkr_concat_condition:
            self.spk_embed_affine_layer = nn.Linear(192, self.spk_embed_dim)
            if self._model_dtype is not None:
                self.spk_embed_affine_layer.to(dtype=self._model_dtype)

        self._vocoder = None
        self._vocoder_sample_rate: int | None = None

        self._stream_cache_lock = threading.Lock()
        self._stream_diff_cache_by_req: dict[str, dict[int | str, Any]] = {}
        self._stream_prompt_payload_by_req: dict[str, dict[str, torch.Tensor]] = {}
        self._stream_audio_tail_by_req: dict[str, torch.Tensor] = {}
        self._stream_wav_pointer_by_req: dict[str, int] = {}
        self._stream_fade_tail_by_req: dict[str, torch.Tensor] = {}
        # Track recently-seen req_ids so stale entries (aborted requests)
        # can be evicted without modifying the shared model runner.
        self._stream_last_seen: dict[str, int] = {}
        self._stream_forward_count: int = 0

        self._flow_weights_path = os.path.join(self.model_dir, "flow", "flow.pt")
        self._hift_weights_path = os.path.join(self.model_dir, "hift", "hift.pt")
        self._vocos_jit_path = os.path.join(self.model_dir, "vocos2d", "generator_jit.ckpt")

    @staticmethod
    def _connector_chunk_config(vllm_config: VllmConfig) -> tuple[int, int]:
        cc = getattr(vllm_config.model_config, "stage_connector_config", None)
        extra = (cc or {}).get("extra") if isinstance(cc, dict) else getattr(cc, "extra", None)
        if not isinstance(extra, dict):
            return 25, 25
        cf = extra.get("codec_chunk_frames", 25)
        return (int(cf[0]) if isinstance(cf, list) else int(cf)), int(extra.get("codec_left_context_frames", 25))

    @staticmethod
    def _resolve_model_dtype(vllm_config: VllmConfig) -> torch.dtype | None:
        dtype = getattr(vllm_config.model_config, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            _map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "half": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            return _map.get(dtype)
        return None

    def _ensure_vocoder_loaded(self) -> None:
        if self._vocoder is not None:
            return

        device = next(self.dit.parameters()).device
        self._vocoder, self._vocoder_sample_rate = load_vocoder(self.model_dir, device, self.sample_rate)
        if self._vocoder is None:
            raise RuntimeError(
                "GLM-TTS stage 1 could not load a vocoder. "
                f"Expected either {self._hift_weights_path} or {self._vocos_jit_path}."
            )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        """Return zero embeddings (DiT does not use token embeddings)."""
        if input_ids.dim() != 1:
            raise ValueError(f"GLM-TTS DiT input_ids must be 1D, got shape={tuple(input_ids.shape)}")
        hidden = int(self.config.hidden_size)
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=self._model_dtype or torch.float32,
        )

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        """DiT does not produce logits for sampling."""
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: Any,
    ) -> OmniOutput:
        """Run DiT flow-matching + vocoder on speech tokens."""
        # Evict stale streaming caches (aborted requests whose stream_finished
        # was never sent).  Entries not seen for 64 forward calls are dropped.
        self._evict_stale_stream_caches()

        runtime_info = kwargs.get("model_intermediate_buffer")
        if runtime_info is None:
            runtime_info = kwargs.get("runtime_additional_information", [])

        seq_token_counts = kwargs.get("seq_token_counts")
        flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
        if seq_token_counts is None and self.max_num_seqs > 1:
            raise RuntimeError(
                "GLM-TTS DiT stage requires seq_token_counts when max_num_seqs > 1; "
                "otherwise concatenated speech tokens cannot be split per request."
            )
        request_ids_list = split_request_ids(flat_ids, seq_token_counts)

        num_reqs = max(1, len(request_ids_list))
        sample_rate = torch.tensor(int(self.sample_rate), dtype=torch.int32)
        empty_audio = torch.zeros((0,), dtype=torch.float32, device=input_ids.device)
        audios: list[torch.Tensor] = [empty_audio] * num_reqs
        srs: list[torch.Tensor] = [sample_rate] * num_reqs

        if not isinstance(runtime_info, list):
            runtime_info = []

        for idx, req_ids in enumerate(request_ids_list):
            info = runtime_info[idx] if idx < len(runtime_info) and isinstance(runtime_info[idx], dict) else {}
            req_id = as_str(info.get("req_id")) if info else None
            if req_id is not None:
                with self._stream_cache_lock:
                    self._stream_last_seen[req_id] = self._stream_forward_count
            stream_finished = as_bool(info.get("stream_finished")) if info else False
            prompt_token = as_tensor(info.get("prompt_speech_token")) if info else None
            if prompt_token is None and info:
                prompt_token = as_tensor(info.get("prompt_token"))  # backward compat
            prompt_feat = as_tensor(info.get("prompt_feat")) if info else None
            embedding = as_tensor(info.get("embedding")) if info else None

            speech_tokens_raw = info.get("speech_tokens") if info else None
            uses_streaming = bool(info) and (
                "stream_finished" in info or "token_offset" in info or "left_context_size" in info
            )
            if uses_streaming and req_id is not None:
                with self._stream_cache_lock:
                    prompt_payload = self._stream_prompt_payload_by_req.setdefault(req_id, {})
                if prompt_token is not None:
                    prompt_payload["prompt_speech_token"] = prompt_token
                elif "prompt_speech_token" in prompt_payload:
                    prompt_token = prompt_payload["prompt_speech_token"]
                if prompt_feat is not None:
                    prompt_payload["prompt_feat"] = prompt_feat
                elif "prompt_feat" in prompt_payload:
                    prompt_feat = prompt_payload["prompt_feat"]
                if embedding is not None:
                    prompt_payload["embedding"] = embedding
                elif "embedding" in prompt_payload:
                    embedding = prompt_payload["embedding"]

            voice_clone_fields = {
                "prompt_speech_token": prompt_token is not None,
                "prompt_feat": prompt_feat is not None,
                "embedding": embedding is not None,
            }
            if any(voice_clone_fields.values()) and not all(voice_clone_fields.values()):
                missing = [name for name, present in voice_clone_fields.items() if not present]
                present = [name for name, present in voice_clone_fields.items() if present]
                raise RuntimeError(
                    "GLM-TTS voice clone payload is incomplete: "
                    f"present={present}, missing={missing}. "
                    "The AR→DiT stage bridge must propagate prompt_speech_token, "
                    "prompt_feat, and embedding together."
                )

            fallback_token = req_ids.reshape(-1).to(device=input_ids.device, dtype=torch.long)
            if uses_streaming:
                token = valid_speech_tokens(None, device=input_ids.device, fallback=fallback_token)
            else:
                token = valid_speech_tokens(
                    speech_tokens_raw,
                    device=input_ids.device,
                    fallback=fallback_token,
                )

            if token.numel() == 0:
                if stream_finished and req_id is not None:
                    with self._stream_cache_lock:
                        self._stream_diff_cache_by_req.pop(req_id, None)
                        self._stream_prompt_payload_by_req.pop(req_id, None)
                        self._stream_audio_tail_by_req.pop(req_id, None)
                        self._stream_wav_pointer_by_req.pop(req_id, None)
                        self._stream_fade_tail_by_req.pop(req_id, None)
                        self._stream_last_seen.pop(req_id, None)
                audios[idx] = empty_audio
                continue

            if uses_streaming:
                token_offset = 0
                try:
                    if info and "token_offset" in info:
                        token_offset = max(0, int(info.get("token_offset", 0)))
                    elif info:
                        token_offset = max(0, int(info.get("left_context_size", 0)))
                except (TypeError, ValueError):
                    token_offset = 0

                chunk_sizes_history = info.get("chunk_sizes_history") if info else None
                block_pattern = info.get("block_pattern") if info else None
                crossfade_sec = float(info.get("crossfade_sec", 0.1)) if info else 0.1

                audio = self._forward_full(
                    speech_tokens=token,
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=embedding,
                    token_offset=token_offset,
                    req_id=req_id,
                    stream_finished=stream_finished,
                    chunk_sizes_history=chunk_sizes_history,
                    block_pattern=block_pattern,
                    crossfade_sec=crossfade_sec,
                )
                if stream_finished and req_id is not None:
                    with self._stream_cache_lock:
                        self._stream_diff_cache_by_req.pop(req_id, None)
                        self._stream_prompt_payload_by_req.pop(req_id, None)
                        self._stream_audio_tail_by_req.pop(req_id, None)
                        self._stream_last_seen.pop(req_id, None)
            else:
                audio = self._forward_full(
                    speech_tokens=token,
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=embedding,
                )

            audios[idx] = audio.reshape(-1).to(dtype=torch.float32)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": audios, "sr": srs},
        )

    @torch.inference_mode()
    def _forward_full(
        self,
        speech_tokens: torch.Tensor,
        prompt_token: torch.Tensor | None = None,
        prompt_feat: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        token_offset: int = 0,
        req_id: str | None = None,
        stream_finished: bool = False,
        chunk_sizes_history: list[int] | None = None,
        block_pattern: list[int] | None = None,
        crossfade_sec: float = 0.1,
    ) -> torch.Tensor:
        """DiT flow-matching + vocoder pipeline. Returns audio waveform."""
        device = speech_tokens.device

        if speech_tokens.ndim == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        generated_token_len = int(speech_tokens.shape[1])

        voice_clone_fields = {
            "prompt_speech_token": prompt_token is not None,
            "prompt_feat": prompt_feat is not None,
            "embedding": embedding is not None,
        }
        if any(voice_clone_fields.values()) and not all(voice_clone_fields.values()):
            missing = [name for name, present in voice_clone_fields.items() if not present]
            present = [name for name, present in voice_clone_fields.items() if present]
            raise RuntimeError(
                "GLM-TTS voice clone payload is incomplete: "
                f"present={present}, missing={missing}. "
                "Refusing to synthesize prompt speech as target audio."
            )

        has_voice_clone = prompt_token is not None or embedding is not None
        if prompt_token is not None:
            if isinstance(prompt_token, np.ndarray):
                prompt_token = torch.from_numpy(prompt_token).long().to(device)
            else:
                prompt_token = prompt_token.to(device=device, dtype=torch.long)
            if prompt_token.ndim == 1:
                prompt_token = prompt_token.unsqueeze(0)
            prompt_token_len = prompt_token.shape[1]
            speech_tokens = torch.cat([prompt_token, speech_tokens], dim=1)
        else:
            prompt_token_len = 0
        has_prompt_audio_context = prompt_token_len > 0

        token_len = speech_tokens.shape[1]
        if token_len == 0:
            return torch.zeros(0, device=device)

        # Real token length for interpolate_token alignment.
        # Must be computed AFTER prompt_token concatenation (above) so that
        # the TextEmbedding sees the actual token count, not the bucket-padded
        # length fed to the CUDA graph static buffers.
        text_lens = torch.tensor([token_len], device=device, dtype=torch.long)

        feat_len = int(token_len / self.input_frame_rate * self.mel_framerate)
        feat_len = max(1, feat_len)

        padding_mask = ~make_pad_mask(torch.tensor([feat_len], device=device), max_len=feat_len)
        mel_cond = torch.zeros([1, feat_len, self.mel_dim], device=device)

        prompt_feat_len = 0
        if prompt_feat is not None:
            if isinstance(prompt_feat, np.ndarray):
                prompt_feat = torch.from_numpy(prompt_feat).to(device=device, dtype=mel_cond.dtype)
            else:
                prompt_feat = prompt_feat.to(device=device, dtype=mel_cond.dtype)
            if prompt_feat.ndim == 2:
                prompt_feat = prompt_feat.unsqueeze(0)
            copy_len = min(prompt_feat.shape[1], feat_len)
            mel_cond[:, :copy_len, :] = prompt_feat[:, -copy_len:, :]
            prompt_feat_len = copy_len
        has_prompt_audio_context = has_prompt_audio_context and prompt_feat_len > 0

        spkr_embedding = None
        if not self.remove_spkr_concat_condition:
            if has_voice_clone and embedding is not None:
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).to(device=device, dtype=mel_cond.dtype)
                else:
                    embedding = embedding.to(device=device, dtype=mel_cond.dtype)
                if embedding.ndim == 1:
                    embedding = embedding.unsqueeze(0)
                spk_embedding_normed = torch.nn.functional.normalize(embedding, dim=1)
                spk_embedding = self.spk_embed_affine_layer(spk_embedding_normed)
                spk_embedding_expanded = spk_embedding.unsqueeze(1).expand(-1, feat_len, -1)
                condition = torch.cat([mel_cond, spk_embedding_expanded], dim=-1)
                spkr_embedding = spk_embedding_normed
            else:
                spk_zeros = torch.zeros([1, feat_len, self.spk_embed_dim], device=device)
                condition = torch.cat([mel_cond, spk_zeros], dim=-1)
        else:
            condition = mel_cond

        if self.spkr_emb_adaln:
            if has_voice_clone and embedding is not None:
                if spkr_embedding is None:
                    if isinstance(embedding, np.ndarray):
                        spkr_embedding = torch.from_numpy(embedding).to(device=device, dtype=mel_cond.dtype)
                    elif isinstance(embedding, torch.Tensor):
                        spkr_embedding = embedding.to(device=device, dtype=mel_cond.dtype)
                    if spkr_embedding is not None and spkr_embedding.ndim == 1:
                        spkr_embedding = spkr_embedding.unsqueeze(0)
                    if spkr_embedding is not None:
                        spkr_embedding = torch.nn.functional.normalize(spkr_embedding, dim=1)
            else:
                spkr_embedding = torch.zeros([1, 192], device=device, dtype=mel_cond.dtype)

        with self._stream_cache_lock:
            last_step_cache = self._stream_diff_cache_by_req.get(req_id) if req_id is not None else None

        dit_block_pattern: list[int] | None = None
        is_streaming = req_id is not None and token_offset >= 0
        if is_streaming:
            pattern_source = block_pattern or chunk_sizes_history
            if pattern_source is not None and len(pattern_source) > 0:
                dit_block_pattern = [prompt_token_len] + pattern_source if prompt_token_len > 0 else list(pattern_source)

        lookahead_mel_len = 0
        if is_streaming and chunk_sizes_history is not None and len(chunk_sizes_history) > 0:
            last_chunk_tokens = chunk_sizes_history[-1]
            lookahead_tokens = last_chunk_tokens // 2
            lookahead_mel_len = int(lookahead_tokens / self.input_frame_rate * self.mel_framerate)

        # Compute cache_len: only cache the stable region for streaming
        stream_cache_len: int | None = None
        if req_id is not None:
            stable_mel_len = int(generated_token_len / self.input_frame_rate * self.mel_framerate)
            override_mel = max(0, stable_mel_len - lookahead_mel_len)
            stream_cache_len = min(feat_len, prompt_feat_len + override_mel + 1)

        mel, current_step_cache = self._do_sample(
            speech_tokens=speech_tokens,
            mel_cond=mel_cond,
            condition=condition,
            padding_mask=padding_mask,
            spkr_embedding=spkr_embedding,
            n_timesteps=self.n_timesteps,
            last_step_cache=last_step_cache,
            block_pattern=dit_block_pattern,
            cache_len=stream_cache_len,
            text_lens=text_lens,
        )

        if req_id is not None:
            current_step_cache["override_len"] = int(stream_cache_len or 0)
            with self._stream_cache_lock:
                self._stream_diff_cache_by_req[req_id] = current_step_cache

        if has_prompt_audio_context:
            crop_len = min(prompt_feat_len, max(0, int(mel.shape[1]) - 1))
            if crop_len > 0:
                mel = mel[:, crop_len:, :]

        self._ensure_vocoder_loaded()

        audio = mel_to_audio(self._vocoder, mel)

        effective_sr = self._vocoder_sample_rate or self.sample_rate
        if effective_sr != self.sample_rate and audio is not None and audio.numel() > 0:
            try:
                import torchaudio

                audio = torchaudio.functional.resample(audio, effective_sr, self.sample_rate)
            except Exception as e:
                logger.warning(
                    "Failed to resample audio from %d to %d: %s",
                    effective_sr,
                    self.sample_rate,
                    e,
                )

        audio = audio.reshape(-1)
        if is_streaming and req_id is not None:
            audio = self._slice_streaming_audio(
                audio,
                req_id=req_id,
                stream_finished=stream_finished,
                lookahead_mel_len=lookahead_mel_len,
                crossfade_sec=crossfade_sec,
            )

        return audio

    def _slice_streaming_audio(
        self,
        audio: torch.Tensor,
        *,
        req_id: str,
        stream_finished: bool,
        lookahead_mel_len: int,
        crossfade_sec: float,
    ) -> torch.Tensor:
        """Slice cumulative streaming audio using the official wav pointer.

        Official GLM-TTS runs the flow/vocoder on the cumulative token prefix
        every chunk, then drops the waveform samples that were already emitted.
        Non-final chunks additionally keep a lookahead plus fade tail for the
        next chunk; the final full-prefix pass flushes that tail.
        """
        look_back_len = max(0, int(lookahead_mel_len * self.sample_rate / self.mel_framerate))
        fade_len = max(0, int(crossfade_sec * self.sample_rate))
        overlap_len = look_back_len + fade_len

        with self._stream_cache_lock:
            wav_pointer = int(self._stream_wav_pointer_by_req.get(req_id, 0))
            fade_tail = self._stream_fade_tail_by_req.get(req_id)

        current = audio[min(wav_pointer, int(audio.numel())) :].clone()
        if fade_tail is not None and fade_tail.numel() > 0 and current.numel() > 0:
            blend_len = min(int(fade_tail.numel()), int(current.numel()))
            fade_in = torch.linspace(0.0, 1.0, blend_len, dtype=current.dtype, device=current.device)
            fade_out = 1.0 - fade_in
            current[:blend_len] = (
                fade_tail[:blend_len].to(device=current.device, dtype=current.dtype) * fade_out
                + current[:blend_len] * fade_in
            )

        if stream_finished:
            with self._stream_cache_lock:
                self._stream_wav_pointer_by_req.pop(req_id, None)
                self._stream_fade_tail_by_req.pop(req_id, None)
            return current

        if overlap_len <= 0 or current.numel() <= overlap_len:
            emit = torch.zeros(0, dtype=current.dtype, device=current.device)
            next_tail = current.detach().clone()
        else:
            emit = current[:-overlap_len]
            tail_end = -look_back_len if look_back_len > 0 else None
            next_tail = current[-overlap_len:tail_end].detach().clone()

        with self._stream_cache_lock:
            self._stream_wav_pointer_by_req[req_id] = wav_pointer + int(emit.numel())
            self._stream_fade_tail_by_req[req_id] = next_tail
        return emit

    def cleanup_request(self, req_id: str) -> None:
        """Remove all streaming state for a request.

        Called when a request finishes normally or is aborted/dropped
        (e.g. client disconnect).  Without this, per-request diffusion
        caches and audio tails leak indefinitely.
        """
        with self._stream_cache_lock:
            self._stream_diff_cache_by_req.pop(req_id, None)
            self._stream_prompt_payload_by_req.pop(req_id, None)
            self._stream_audio_tail_by_req.pop(req_id, None)
            self._stream_wav_pointer_by_req.pop(req_id, None)
            self._stream_fade_tail_by_req.pop(req_id, None)
            self._stream_last_seen.pop(req_id, None)

    # Maximum number of forward calls before a cache entry is considered stale.
    _STALE_THRESHOLD = 64

    def _evict_stale_stream_caches(self) -> None:
        """Evict streaming cache entries for requests not seen recently.

        This is a self-contained fallback: if ``stream_finished`` is never
        sent (e.g. client disconnect), the entry would otherwise leak forever.
        Called at the top of every ``forward`` so no changes to the shared
        model runner are needed.
        """
        self._stream_forward_count += 1
        # Only check every 16 forward calls to amortize the lock overhead.
        if self._stream_forward_count % 16 != 0:
            return
        with self._stream_cache_lock:
            if not self._stream_last_seen:
                return
            stale_ids = [
                rid
                for rid, last_seen in self._stream_last_seen.items()
                if self._stream_forward_count - last_seen > self._STALE_THRESHOLD
            ]
            for rid in stale_ids:
                self._stream_diff_cache_by_req.pop(rid, None)
                self._stream_prompt_payload_by_req.pop(rid, None)
                self._stream_audio_tail_by_req.pop(rid, None)
                self._stream_wav_pointer_by_req.pop(rid, None)
                self._stream_fade_tail_by_req.pop(rid, None)
                self._stream_last_seen.pop(rid, None)

    def _do_sample(
        self,
        speech_tokens: torch.Tensor,
        mel_cond: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        spkr_embedding: torch.Tensor | None,
        n_timesteps: int,
        last_step_cache: dict[int | str, Any] | None = None,
        block_pattern: list[int] | None = None,
        cache_len: int | None = None,
        text_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int | str, Any]]:
        """Euler ODE sampling for flow matching."""
        # CUDA graph path: the captured graph does NOT pass is_causal /
        # block_pattern to the DiT, so it always uses bidirectional attention.
        # That is correct for non-streaming (full-sequence) inference but
        # wrong for streaming where block-causal attention is required.
        # Skip the graph when block_pattern is set so we fall through to the
        # eager path that correctly forwards these arguments.
        use_cudagraph = block_pattern is None
        graph_result = (
            self._try_cudagraph_sample(
                speech_tokens=speech_tokens,
                mel_cond=mel_cond,
                condition=condition,
                padding_mask=padding_mask,
                spkr_embedding=spkr_embedding,
                n_timesteps=n_timesteps,
                last_step_cache=last_step_cache,
                cache_len=cache_len,
                text_lens=text_lens,
            )
            if use_cudagraph
            else None
        )
        if graph_result is not None:
            return graph_result

        device = speech_tokens.device
        # Always accumulate the Euler ODE state in float32 regardless of the
        # DiT model dtype.  Half-precision rounding errors across 10 steps (×2
        # CFG passes each) manifest as audible hiss on H20/H100.
        x = torch.randn(mel_cond.shape, device=device, dtype=torch.float32)
        # Cast conditioning tensors to float32 for the ODE loop.
        condition_f32 = condition.to(torch.float32)
        spkr_embedding_f32 = spkr_embedding.to(torch.float32) if spkr_embedding is not None else None
        current_step_cache: dict[int | str, Any] = {}
        effective_cache_len = min(int(cache_len or 0), int(mel_cond.shape[1]))
        should_cache_steps = effective_cache_len > 0
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=torch.float32)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        for step in range(n_timesteps):
            cache_step = step + 1
            if last_step_cache is not None and cache_step in last_step_cache:
                x_cache = last_step_cache[cache_step].get("x")
                if isinstance(x_cache, torch.Tensor):
                    override_len = int(last_step_cache.get("override_len", x_cache.shape[1]))
                    safe_len = min(int(x.shape[1]), int(x_cache.shape[1]), max(0, override_len))
                    if safe_len > 0:
                        x[:, :safe_len, :] = x_cache[:, :safe_len, :].to(device=device, dtype=torch.float32)
            if should_cache_steps:
                current_step_cache[cache_step] = {"x": x[:, :effective_cache_len, :].detach().clone()}

            t_current = t_span[step]
            dt = t_span[step + 1] - t_current

            use_causal = block_pattern is not None
            dphi_dt = self.dit(
                middle_point=x.to(mel_cond.dtype),
                condition=condition_f32.to(mel_cond.dtype),
                text=speech_tokens,
                time_step=t_current.unsqueeze(0),
                padding_mask=padding_mask,
                spkr_emb=spkr_embedding_f32.to(mel_cond.dtype)
                if spkr_embedding_f32 is not None and self.spkr_emb_adaln
                else None,
                is_causal=use_causal,
                block_pattern=block_pattern,
                text_lens=text_lens,
            ).to(torch.float32)

            if self.inference_cfg_rate > 0:
                text_uncond = torch.zeros_like(speech_tokens) if self.speech_token_cfg else speech_tokens
                spkr_uncond = (
                    torch.zeros_like(spkr_embedding_f32).to(mel_cond.dtype) if spkr_embedding_f32 is not None else None
                )
                cfg_dphi_dt = self.dit(
                    middle_point=x.to(mel_cond.dtype),
                    condition=torch.zeros_like(condition_f32).to(mel_cond.dtype),
                    text=text_uncond,
                    time_step=t_current.unsqueeze(0),
                    padding_mask=padding_mask,
                    spkr_emb=spkr_uncond if self.spkr_emb_adaln else None,
                    is_causal=use_causal,
                    block_pattern=block_pattern,
                    text_lens=text_lens,
                ).to(torch.float32)
                dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt

            x = x + dt * dphi_dt

        return x.to(mel_cond.dtype), current_step_cache

    def _try_cudagraph_sample(
        self,
        *,
        speech_tokens: torch.Tensor,
        mel_cond: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        spkr_embedding: torch.Tensor | None,
        n_timesteps: int,
        last_step_cache: dict[int | str, Any] | None,
        cache_len: int | None = None,
        text_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int | str, Any]] | None:
        if not self._use_dit_cuda_graphs or mel_cond.device.type != "cuda":
            return None
        if self._dit_cudagraph is None:
            condition_dim = self.mel_dim if self.remove_spkr_concat_condition else self.mel_dim + self.spk_embed_dim
            self._dit_cudagraph = CUDAGraphGLMTTSDiTWrapper(
                self.dit,
                mel_dim=self.mel_dim,
                condition_dim=condition_dim,
                input_frame_rate=self.input_frame_rate,
                mel_framerate=self.mel_framerate,
                inference_cfg_rate=self.inference_cfg_rate,
                speech_token_cfg=self.speech_token_cfg,
                spkr_emb_adaln=self.spkr_emb_adaln,
                t_scheduler=self.t_scheduler,
                enabled=True,
            )
            try:
                self._dit_cudagraph.warmup(
                    device=mel_cond.device,
                    dtype=mel_cond.dtype,
                    n_timesteps=n_timesteps,
                    codec_chunk_frames=self._codec_chunk_frames,
                    codec_left_context_frames=self._codec_left_context_frames,
                )
            except Exception:
                logger.warning("Disabling GLM-TTS DiT CUDA graphs after warmup failure", exc_info=True)
                self._dit_cudagraph.enabled = False
                return None

        try:
            return self._dit_cudagraph.sample(
                speech_tokens=speech_tokens,
                mel_cond=mel_cond,
                condition=condition,
                padding_mask=padding_mask,
                spkr_embedding=spkr_embedding,
                n_timesteps=n_timesteps,
                last_step_cache=last_step_cache,
                cache_len=cache_len,
                text_lens=text_lens,
            )
        except Exception:
            logger.warning("GLM-TTS DiT CUDA graph replay failed; falling back to eager", exc_info=True)
            return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load DiT + vocoder weights from {model_dir}/flow/flow.pt."""
        for _ in weights:
            pass

        weight_dict: dict[str, torch.Tensor] = {}
        if not os.path.exists(self._flow_weights_path):
            raise FileNotFoundError(f"GLM-TTS flow weights not found at {self._flow_weights_path}")

        logger.info("Loading GLM-TTS flow weights from %s", self._flow_weights_path)
        checkpoint = torch.load(self._flow_weights_path, map_location="cpu", weights_only=True)
        if "model" in checkpoint:
            weight_dict = checkpoint["model"]
        else:
            weight_dict = checkpoint

        dit_state = {
            (name.removeprefix("estimator.") if name.startswith("estimator.") else name.removeprefix("dit.")): tensor
            for name, tensor in weight_dict.items()
            if (name.startswith("estimator.") or name.startswith("dit."))
            and "inv_freq" not in name
            and "rotary_embed" not in name
        }
        if not dit_state:
            raise RuntimeError(
                "GLM-TTS flow checkpoint did not contain any DiT weights under 'estimator.*' or 'dit.*'."
            )
        if dit_state:
            missing, unexpected = self.dit.load_state_dict(dit_state, strict=False)
            missing = [k for k in missing if "inv_freq" not in k]
            if missing:
                logger.warning("Missing DiT weights: %s", missing[:10])
            if unexpected:
                logger.warning("Unexpected DiT weights: %s", unexpected[:10])
            logger.info("Loaded %d DiT weights", len(dit_state))

        loaded_spk: set[str] = set()
        if hasattr(self, "spk_embed_affine_layer"):
            for suffix in ("weight", "bias"):
                key = f"spk_embed_affine_layer.{suffix}"
                if key in weight_dict:
                    param = getattr(self.spk_embed_affine_layer, suffix)
                    param.data.copy_(weight_dict[key].to(device=param.device, dtype=param.dtype))
                    loaded_spk.add(key)
            self.spk_embed_affine_layer.eval()
        self.dit.eval()
        self._ensure_vocoder_loaded()

        # Return both prefixed and unprefixed param names for nested module compat
        loaded_params: set[str] = set()
        for k in dit_state:
            loaded_params.update((f"dit.{k}", f"_dit_gen.dit.{k}"))
        for k in loaded_spk:
            loaded_params.update((k, f"_dit_gen.{k}"))
        logger.info("Loaded GLMTTSDiTForGeneration: %d params", len(loaded_params))
        return loaded_params


class CUDAGraphGLMTTSDiTWrapper:
    """Capture GLM-TTS DiT Euler sampling for fixed mel-frame buckets.

    The wrapper owns static buffers for a single-request DiT stage. Dynamic
    requests are right-padded to the nearest captured mel-frame bucket and
    trimmed back to the actual mel length after graph replay.
    """

    def __init__(
        self,
        dit: torch.nn.Module,
        *,
        mel_dim: int,
        condition_dim: int,
        input_frame_rate: float,
        mel_framerate: float,
        inference_cfg_rate: float,
        speech_token_cfg: bool,
        spkr_emb_adaln: bool,
        t_scheduler: str,
        enabled: bool = True,
        capture_sizes: list[int] | None = None,
    ) -> None:
        self.dit = dit
        self.mel_dim = int(mel_dim)
        self.condition_dim = int(condition_dim)
        self.input_frame_rate = float(input_frame_rate)
        self.mel_framerate = float(mel_framerate)
        self.inference_cfg_rate = float(inference_cfg_rate)
        self.speech_token_cfg = bool(speech_token_cfg)
        self.spkr_emb_adaln = bool(spkr_emb_adaln)
        self.t_scheduler = str(t_scheduler)
        self.enabled = bool(enabled)
        self.capture_sizes = sorted(set(capture_sizes or []))

        self.graphs: dict[int, CUDAGraph] = {}
        self.static_x: dict[int, torch.Tensor] = {}
        self.static_condition: dict[int, torch.Tensor] = {}
        self.static_text: dict[int, torch.Tensor] = {}
        self.static_text_lens: dict[int, torch.Tensor] = {}
        self.static_padding_mask: dict[int, torch.Tensor] = {}
        self.static_spkr_emb: dict[int, torch.Tensor] = {}
        self.static_output: dict[int, torch.Tensor] = {}
        self.static_step_cache: dict[int, list[torch.Tensor]] = {}

        self._warmed_up = False
        self._n_timesteps: int | None = None

    @staticmethod
    def compute_capture_sizes(
        *,
        codec_chunk_frames: int = 25,
        codec_left_context_frames: int = 25,
        input_frame_rate: float = 25.0,
        mel_framerate: float = 50.0,
        max_bucket: int = 1024,
    ) -> list[int]:
        sizes: set[int] = set()

        def token_to_mel(tokens: int) -> int:
            return max(1, int(math.ceil(float(tokens) / input_frame_rate * mel_framerate)))

        if codec_chunk_frames > 0:
            sizes.add(token_to_mel(codec_chunk_frames))
            if codec_left_context_frames > 0:
                sizes.add(token_to_mel(codec_chunk_frames + codec_left_context_frames))

        for size in (32, 64, 128, 256, 512, 1024):
            if size <= max_bucket:
                sizes.add(size)
        return sorted(sizes)

    def _get_padded_size(self, actual_feat_len: int) -> int | None:
        for size in self.capture_sizes:
            if actual_feat_len <= size:
                return size
        return None

    def _token_bucket_len(self, bucket_feat_len: int) -> int:
        return max(1, int(math.ceil(float(bucket_feat_len) * self.input_frame_rate / self.mel_framerate)))

    def warmup(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_timesteps: int,
        codec_chunk_frames: int = 25,
        codec_left_context_frames: int = 25,
    ) -> None:
        if not self.enabled or self._warmed_up or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        self._n_timesteps = int(n_timesteps)
        if not self.capture_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
                input_frame_rate=self.input_frame_rate,
                mel_framerate=self.mel_framerate,
            )

        self.dit.eval()
        logger.info("Starting GLM-TTS DiT CUDA graph warmup for sizes: %s", self.capture_sizes)
        for size in self.capture_sizes:
            try:
                self._allocate_static_buffers(size, device, dtype)
                self._capture(size, device)
                logger.info("Captured GLM-TTS DiT CUDA graph for mel frames=%d", size)
            except Exception:
                logger.warning("Failed to capture GLM-TTS DiT CUDA graph for size=%d", size, exc_info=True)
        self._warmed_up = True

    def _allocate_static_buffers(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        token_len = self._token_bucket_len(size)
        self.static_x[size] = torch.zeros(1, size, self.mel_dim, device=device, dtype=dtype)
        self.static_condition[size] = torch.zeros(1, size, self.condition_dim, device=device, dtype=dtype)
        self.static_text[size] = torch.zeros(1, token_len, device=device, dtype=torch.long)
        self.static_text_lens[size] = torch.full((1,), token_len, device=device, dtype=torch.long)
        self.static_padding_mask[size] = torch.zeros(1, size, device=device, dtype=torch.bool)
        self.static_spkr_emb[size] = torch.zeros(1, 192, device=device, dtype=dtype)
        self.static_output[size] = torch.zeros(1, size, self.mel_dim, device=device, dtype=dtype)
        self.static_step_cache[size] = [
            torch.zeros(1, size, self.mel_dim, device=device, dtype=dtype) for _ in range(int(self._n_timesteps or 0))
        ]

    def _time_schedule(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_timesteps = int(self._n_timesteps or 0)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return t_span

    def _run_static(self, size: int) -> torch.Tensor:
        x = self.static_x[size]
        condition = self.static_condition[size]
        text = self.static_text[size]
        text_lens = self.static_text_lens[size]
        padding_mask = self.static_padding_mask[size]
        spkr_emb = self.static_spkr_emb[size] if self.spkr_emb_adaln else None
        t_span = self._time_schedule(x.device, x.dtype)

        for step in range(int(self._n_timesteps or 0)):
            self.static_step_cache[size][step].copy_(x)
            t_current = t_span[step]
            dt = t_span[step + 1] - t_current
            dphi_dt = self.dit(
                middle_point=x,
                condition=condition,
                text=text,
                text_lens=text_lens,
                time_step=t_current.unsqueeze(0),
                padding_mask=padding_mask,
                spkr_emb=spkr_emb,
            )
            if self.inference_cfg_rate > 0:
                text_uncond = torch.zeros_like(text) if self.speech_token_cfg else text
                spkr_uncond = torch.zeros_like(spkr_emb) if spkr_emb is not None else None
                cfg_dphi_dt = self.dit(
                    middle_point=x,
                    condition=torch.zeros_like(condition),
                    text=text_uncond,
                    text_lens=text_lens,
                    time_step=t_current.unsqueeze(0),
                    padding_mask=padding_mask,
                    spkr_emb=spkr_uncond if self.spkr_emb_adaln else None,
                )
                dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
        self.static_output[size].copy_(x)
        return self.static_output[size]

    def _capture(self, size: int, device: torch.device) -> None:
        with torch.no_grad():
            self.static_x[size].normal_()
            _ = self._run_static(size)
        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                self._run_static(size)
        self.graphs[size] = graph

    def sample(
        self,
        *,
        speech_tokens: torch.Tensor,
        mel_cond: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        spkr_embedding: torch.Tensor | None,
        n_timesteps: int,
        last_step_cache: dict[int | str, Any] | None = None,
        cache_len: int | None = None,
        text_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int | str, Any]] | None:
        if (
            not self.enabled
            or not self._warmed_up
            or int(n_timesteps) != int(self._n_timesteps or -1)
            or speech_tokens.shape[0] != 1
            or mel_cond.shape[0] != 1
            or mel_cond.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            return None

        actual_feat_len = int(mel_cond.shape[1])
        size = self._get_padded_size(actual_feat_len)
        if size is None or size not in self.graphs:
            return None

        token_bucket_len = int(self.static_text[size].shape[1])
        if int(speech_tokens.shape[1]) > token_bucket_len:
            return None

        self.static_x[size].normal_()
        self.static_condition[size].zero_()
        self.static_text[size].zero_()
        self.static_padding_mask[size].zero_()
        self.static_spkr_emb[size].zero_()

        # Warm-start: when streaming, pre-fill the stable region of the
        # initial noise from the step-1 cache of the previous chunk.
        # The graph's Euler loop then evolves from this warm-started state.
        # New (lookahead) region keeps random noise — no per-step overlay.
        if last_step_cache is not None and 1 in last_step_cache:
            cached_x = last_step_cache[1]["x"]
            override_len = int(last_step_cache.get("override_len", cached_x.shape[1]))
            safe = min(size, int(cached_x.shape[1]), max(0, override_len))
            if safe > 0:
                self.static_x[size][:, :safe, :].copy_(cached_x[:, :safe, :].to(self.static_x[size]))

        self.static_condition[size][:, :actual_feat_len, : condition.shape[-1]] = condition
        self.static_text[size][:, : speech_tokens.shape[1]] = speech_tokens
        actual_text_len = int(speech_tokens.shape[1])
        if text_lens is not None:
            self.static_text_lens[size].copy_(text_lens)
        else:
            self.static_text_lens[size].fill_(actual_text_len)
        self.static_padding_mask[size][:, :actual_feat_len] = padding_mask
        if spkr_embedding is not None:
            self.static_spkr_emb[size].copy_(spkr_embedding.to(self.static_spkr_emb[size]))

        self.graphs[size].replay()
        mel = self.static_output[size][:, :actual_feat_len, :].clone()
        current_step_cache: dict[int | str, Any] = {}
        cache_len = min(int(cache_len or 0), actual_feat_len)
        if cache_len > 0:
            for idx, cache in enumerate(self.static_step_cache[size], start=1):
                current_step_cache[idx] = {"x": cache[:, :cache_len, :].detach().clone()}
        return mel, current_step_cache
