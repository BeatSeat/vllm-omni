# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS AR Model (Stage 0): Text → Speech Tokens.

Based on Llama architecture, generates speech token sequences from input text.
Analogous to Fish Speech Slow AR and Qwen3-TTS Talker models.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.common.tts_sampling import ras_sample_one as _ras_sample_one
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_glm_tts import GLMTTSConfig
from .sampling import (
    req_float,
)
from .text_frontend import GLMTTSTextFrontend
from .voice_clone import (
    extract_prompt_feat,
    extract_prompt_speech_token,
    extract_spk_embedding,
    load_voice_clone_frontend,
)

logger = init_logger(__name__)

_GLM_TTS_DEFAULT_REPO_ID = "zai-org/GLM-TTS"
_GLM_TTS_TOKENIZER_SUBDIR = "vq32k-phoneme-tokenizer"


def is_glm_tts_model_name(model_name: Any) -> bool:
    model_name_str = str(model_name)
    return "GLM-TTS" in model_name_str or "glm_tts" in model_name_str.lower() or "glm-tts" in model_name_str.lower()


def _infer_glm_tts_repo_id_from_path(model_name_or_path: Any) -> str | None:
    model_name_str = str(model_name_or_path)
    if model_name_str == _GLM_TTS_DEFAULT_REPO_ID:
        return _GLM_TTS_DEFAULT_REPO_ID
    for part in os.path.normpath(model_name_str).split(os.sep):
        if part.startswith("models--") and "GLM-TTS" in part:
            return part.removeprefix("models--").replace("--", "/")
    if not os.path.exists(os.fspath(model_name_or_path)):
        return model_name_str if is_glm_tts_model_name(model_name_str) else None
    return _GLM_TTS_DEFAULT_REPO_ID if is_glm_tts_model_name(model_name_str) else None


def resolve_glm_tts_tokenizer_path(model_name_or_path: Any) -> Any:
    model_path = os.fspath(model_name_or_path)
    tokenizer_path: Any = model_name_or_path
    if os.path.exists(model_path):
        candidates = [
            os.path.join(model_path, "vq32k-phoneme-tokenizer"),
        ]
        if os.path.basename(os.path.normpath(model_path)) == "llm":
            candidates.append(os.path.join(os.path.dirname(model_path), "vq32k-phoneme-tokenizer"))
        for candidate in candidates:
            if os.path.isdir(candidate):
                tokenizer_path = candidate
                break
    return tokenizer_path


def _glm_tts_cached_file(model_name_or_path: Any, filename: str) -> str | None:
    try:
        return cached_file(model_name_or_path, filename)
    except Exception:
        logger.debug("cached_file could not resolve %s from %s", filename, model_name_or_path, exc_info=True)
        return None


def _glm_tts_root_from_file(path: str, filename: str) -> str:
    root = os.path.abspath(path)
    for _ in filename.split("/"):
        root = os.path.dirname(root)
    return root


def _glm_tts_root_from_tokenizer_path(tokenizer_path: Any) -> str | None:
    if not tokenizer_path:
        return None
    path = os.fspath(tokenizer_path)
    if not os.path.isdir(path):
        return None
    norm = os.path.abspath(os.path.normpath(path))
    if os.path.basename(norm) == _GLM_TTS_TOKENIZER_SUBDIR:
        return os.path.dirname(norm)
    candidate = os.path.join(norm, _GLM_TTS_TOKENIZER_SUBDIR)
    if os.path.isdir(candidate):
        return norm
    return None


def _glm_tts_root_has_files(root: str, filenames: Sequence[str]) -> bool:
    return all(os.path.isfile(os.path.join(root, *name.split("/"))) for name in filenames)


def _glm_tts_root_has_one_file(root: str, filenames: Sequence[str]) -> bool:
    return not filenames or any(os.path.isfile(os.path.join(root, *name.split("/"))) for name in filenames)


def resolve_glm_tts_model_dir(
    model_name_or_path: Any,
    *,
    tokenizer_path: Any | None = None,
    required_files: Sequence[str] = (),
    optional_files: Sequence[str] = (),
) -> str:
    """Resolve a GLM-TTS repo root while avoiding full snapshot queries.

    The engine often already knows a local tokenizer subdirectory.  Prefer that
    snapshot root, then follow Qwen3-TTS's pattern of resolving individual
    files via ``cached_file``.  Fall back to ``snapshot_download`` only when the
    specific resources cannot identify a usable local root.
    """
    model_path = os.fspath(model_name_or_path)
    if os.path.isdir(model_path):
        return os.path.abspath(model_path)

    tokenizer_root = _glm_tts_root_from_tokenizer_path(tokenizer_path)
    if (
        tokenizer_root is not None
        and _glm_tts_root_has_files(tokenizer_root, required_files)
        and _glm_tts_root_has_one_file(tokenizer_root, optional_files)
    ):
        return tokenizer_root

    resolved_files: list[tuple[str, str]] = []
    for filename in tuple(required_files) + tuple(optional_files):
        resolved = _glm_tts_cached_file(model_name_or_path, filename)
        if resolved is not None:
            resolved_files.append((filename, resolved))

    for filename, resolved in resolved_files:
        root = _glm_tts_root_from_file(resolved, filename)
        if _glm_tts_root_has_files(root, required_files) and _glm_tts_root_has_one_file(root, optional_files):
            return root

    if tokenizer_root is not None and not required_files and not optional_files:
        return tokenizer_root

    from huggingface_hub import snapshot_download

    return snapshot_download(model_name_or_path)


def _first_glm_tts_value(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _glm_tts_int_value(value: Any) -> int | None:
    """Extract a scalar integer from vLLM multimodal/runtime wrappers."""
    value = _first_glm_tts_value(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _decode_glm_tts_audio_data(value: Any) -> tuple[torch.Tensor | None, int | None]:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, torch.Tensor):
        return value, None
    if isinstance(value, tuple) and len(value) == 2:
        wav, sr = value
        return torch.as_tensor(wav), int(sr)
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], (int, float)):
        wav, sr = value
        return torch.as_tensor(wav), int(sr)
    if hasattr(value, "shape"):
        return torch.as_tensor(value), None
    if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
        return torch.as_tensor(value), None
    if isinstance(value, str):
        audio_obj: Any
        if value.startswith("data:"):
            _, _, encoded = value.partition(",")
            audio_obj = io.BytesIO(base64.b64decode(encoded))
        elif os.path.isfile(value):
            audio_obj = value
        else:
            return None, None
        import soundfile as sf

        wav_np, sr = sf.read(audio_obj, dtype="float32")
        if getattr(wav_np, "ndim", 1) > 1:
            wav_np = wav_np.mean(axis=1)
        return torch.from_numpy(wav_np), int(sr)
    return None, None


def load_glm_tts_tokenizer(
    tokenizer_path: Any,
    *,
    model_name_or_path: Any | None = None,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    base_kwargs = dict(kwargs)
    base_kwargs["trust_remote_code"] = trust_remote_code
    attempts: list[tuple[Any, dict[str, Any]]] = []

    def _add_attempt(path: Any, extra_kwargs: dict[str, Any]) -> None:
        key = (str(path), tuple(sorted(extra_kwargs.items())))
        if key not in seen:
            seen.add(key)
            attempts.append((path, extra_kwargs))

    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    _add_attempt(tokenizer_path, {"use_fast": False})
    _add_attempt(tokenizer_path, {})

    if model_name_or_path is not None:
        repo_id = _infer_glm_tts_repo_id_from_path(model_name_or_path)
        if repo_id is not None and not os.path.isdir(os.fspath(tokenizer_path)):
            try:
                from huggingface_hub import snapshot_download

                local_root = snapshot_download(
                    repo_id,
                    allow_patterns=[
                        f"{_GLM_TTS_TOKENIZER_SUBDIR}/tokenizer*",
                        f"{_GLM_TTS_TOKENIZER_SUBDIR}/tokenization*",
                        f"{_GLM_TTS_TOKENIZER_SUBDIR}/special_tokens*",
                        f"{_GLM_TTS_TOKENIZER_SUBDIR}/vocab*",
                        f"{_GLM_TTS_TOKENIZER_SUBDIR}/merges*",
                        f"{_GLM_TTS_TOKENIZER_SUBDIR}/added_tokens*",
                    ],
                )
                local_tokenizer_path = os.path.join(local_root, _GLM_TTS_TOKENIZER_SUBDIR)
                if os.path.isdir(local_tokenizer_path):
                    _add_attempt(local_tokenizer_path, {"use_fast": False})
                    _add_attempt(local_tokenizer_path, {})
            except Exception:
                logger.debug("Failed to pre-download GLM-TTS tokenizer subfolder", exc_info=True)
        if repo_id is not None:
            _add_attempt(repo_id, {"subfolder": "vq32k-phoneme-tokenizer", "use_fast": False})
            _add_attempt(repo_id, {"subfolder": "vq32k-phoneme-tokenizer"})
        if repo_id != _GLM_TTS_DEFAULT_REPO_ID and is_glm_tts_model_name(model_name_or_path):
            _add_attempt(_GLM_TTS_DEFAULT_REPO_ID, {"subfolder": "vq32k-phoneme-tokenizer", "use_fast": False})
            _add_attempt(_GLM_TTS_DEFAULT_REPO_ID, {"subfolder": "vq32k-phoneme-tokenizer"})
        _add_attempt(model_name_or_path, {"subfolder": "vq32k-phoneme-tokenizer", "use_fast": False})
        _add_attempt(model_name_or_path, {"subfolder": "vq32k-phoneme-tokenizer"})
        _add_attempt(model_name_or_path, {"use_fast": False})
        _add_attempt(model_name_or_path, {})

    last_exc: Exception | None = None
    for path, extra_kwargs in attempts:
        call_kwargs = base_kwargs.copy()
        call_kwargs.update(extra_kwargs)
        try:
            return AutoTokenizer.from_pretrained(path, **call_kwargs)
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise ValueError("No GLM-TTS tokenizer load attempts were configured.")


def get_glm_tts_special_token_ids(tokenizer: Any) -> dict[str, int]:
    """Resolve GLM-TTS special IDs from the phoneme/audio tokenizer."""
    special_tokens = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "<|endoftext|>",
    }

    result: dict[str, int] = {}
    for key, token_str in special_tokens.items():
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Token '{key}' ({token_str}) should encode to single ID, got: {token_ids}")
        result[key] = int(token_ids[0])
    return result


def resolve_glm_tts_campplus_path(model_dir: str) -> str | None:
    """Resolve ``campplus.onnx`` for GLM-TTS voice cloning."""
    local = os.path.join(model_dir, "frontend", "campplus.onnx")
    if os.path.isfile(local):
        return local

    try:
        resolved = cached_file("FunAudioLLM/CosyVoice-300M", "campplus.onnx")
        if resolved is not None:
            logger.info("Resolved campplus.onnx from FunAudioLLM/CosyVoice-300M: %s", resolved)
            return resolved
    except Exception:
        logger.debug("cached_file could not fetch campplus.onnx", exc_info=True)

    logger.warning(
        "campplus.onnx not found locally or could not be downloaded. "
        "Voice cloning speaker embedding will not be available.",
    )
    return None


def _normalize_glm_tts_processor_text(
    frontend: GLMTTSTextFrontend,
    value: str | None,
    *,
    add_trailing_space: bool = False,
) -> str:
    if value is None:
        return ""
    normalized = frontend.text_normalize(value)
    normalized = (normalized or value).strip()
    if add_trailing_space and normalized:
        normalized = f"{normalized} "
    return normalized


class GLMTTSMultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(GLMTTSConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=24000,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class GLMTTSMultiModalProcessor(BaseMultiModalProcessor[GLMTTSMultiModalProcessingInfo]):
    """GLM-TTS voice-clone processor.

    Unlike CosyVoice3, GLM-TTS prompt speech tokens are normal Llama vocab IDs
    (``<|audio_N|>``).  The processor exposes them as multimodal embeddings for
    the AR prompt and also carries WhisperVQ/CampPlus outputs to the AR->DiT
    handoff.
    """

    def _ensure_cached_runtime_components(self, model_dir: str, config: GLMTTSConfig) -> None:
        requested_model_dir = model_dir
        tokenizer_path = getattr(self.info.ctx.model_config, "tokenizer", None)
        cached_model_source = getattr(self, "_cached_model_source", None)
        cached_model_dir = getattr(self, "_cached_model_dir", None)
        if cached_model_source == requested_model_dir and cached_model_dir is not None:
            return

        model_dir = resolve_glm_tts_model_dir(
            model_dir,
            tokenizer_path=tokenizer_path,
            required_files=(
                "speech_tokenizer/config.json",
                "speech_tokenizer/model.safetensors",
                "speech_tokenizer/preprocessor_config.json",
            ),
        )

        if cached_model_dir == model_dir:
            self._cached_model_source = requested_model_dir
            return

        if not tokenizer_path or not os.path.isdir(os.fspath(tokenizer_path)):
            tokenizer_path = os.path.join(model_dir, _GLM_TTS_TOKENIZER_SUBDIR)
            if not os.path.isdir(tokenizer_path):
                tokenizer_path = model_dir

        trust_remote_code = bool(getattr(self.info.ctx.model_config, "trust_remote_code", True))
        self.tokenizer = load_glm_tts_tokenizer(
            tokenizer_path,
            model_name_or_path=self.info.ctx.model_config.model,
            trust_remote_code=trust_remote_code,
        )
        self.special_ids = get_glm_tts_special_token_ids(self.tokenizer)
        self.text_frontend = GLMTTSTextFrontend()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor_device = device
        speech_tokenizer_cache = getattr(self, "speech_tokenizer", None)
        campplus_cache = getattr(self, "campplus_session", None)
        campplus_path = getattr(self, "_campplus_path", None)
        if campplus_cache is None and campplus_path is None:
            campplus_path = resolve_glm_tts_campplus_path(model_dir)
            self._campplus_path = campplus_path
        self.speech_tokenizer, self.campplus_session = load_voice_clone_frontend(
            model_dir,
            device,
            speech_tokenizer_cache=speech_tokenizer_cache,
            campplus_cache=campplus_cache,
            campplus_path=campplus_path,
        )
        self._cached_model_dir = model_dir
        self._cached_model_source = requested_model_dir

    def _encode_text(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text)
        return torch.tensor([token_ids], dtype=torch.long)

    def _get_audio(self, mm_data: Mapping[str, object]) -> tuple[torch.Tensor | None, int | None]:
        audio = mm_data.get("audio")
        if audio is None:
            audios = mm_data.get("audios")
            if isinstance(audios, (list, tuple)) and audios:
                # Match CosyVoice3's compatibility behavior and accept the
                # first item from the legacy plural field name.
                audio = audios[0]
            else:
                audio = audios
        wav, sr = _decode_glm_tts_audio_data(audio)
        if wav is not None and wav.ndim > 1:
            wav = wav.float()
            wav = wav.mean(dim=0) if wav.shape[0] <= wav.shape[-1] else wav.mean(dim=-1)
        return wav, int(sr or 24000) if wav is not None else sr

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        self._ensure_cached_runtime_components(model_dir, config)

        normalized_text = _normalize_glm_tts_processor_text(self.text_frontend, prompt)
        text_ids = self._encode_text(normalized_text)
        wav, sr = self._get_audio(mm_data)
        if wav is None:
            audio_supplied = mm_data.get("audio") is not None or mm_data.get("audios") is not None
            assert not audio_supplied, (
                "GLM-TTS received an audio multimodal payload that could not be decoded. "
                "Runtime voice cloning must provide decodable multi_modal_data['audio']; "
                "the text-only processor path is reserved for profiling/cache."
            )
            # Keep parity with CosyVoice3 for vLLM profiling/cache setup,
            # which may call the processor without an audio item. This is not
            # a supported runtime inference mode for GLM-TTS.
            return BatchFeature(
                {
                    "input_ids": text_ids,
                    "input_len": torch.tensor([int(text_ids.shape[1])], dtype=torch.long),
                }
            )

        prompt_text = mm_kwargs.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError("GLM-TTS voice cloning requires mm_processor_kwargs['prompt_text'].")

        normalized_prompt_text = _normalize_glm_tts_processor_text(
            self.text_frontend,
            prompt_text,
            add_trailing_space=True,
        )
        prompt_text_ids = self._encode_text(normalized_prompt_text)

        prompt_speech_token = extract_prompt_speech_token(wav, int(sr or 24000), self.speech_tokenizer)
        if not prompt_speech_token:
            raise RuntimeError("GLM-TTS failed to extract WhisperVQ prompt speech tokens from ref_audio.")

        prompt_speech_token_tensor = torch.tensor([prompt_speech_token], dtype=torch.long)
        boa_tensor = torch.tensor([[int(self.special_ids["boa"])]], dtype=torch.long)
        input_ids = torch.cat([prompt_text_ids, text_ids, boa_tensor], dim=1)
        logger.info(
            "GLM-TTS processor prompt: prompt_text_tokens=%d text_tokens=%d "
            "prompt_speech_tokens=%d input_tokens_before_audio=%d expected_prefill_tokens=%d",
            int(prompt_text_ids.shape[1]),
            int(text_ids.shape[1]),
            len(prompt_speech_token),
            int(input_ids.shape[1]),
            int(input_ids.shape[1]) + len(prompt_speech_token),
        )

        prompt_feat = extract_prompt_feat(wav, int(sr or 24000), self.processor_device)
        if prompt_feat is None:
            raise RuntimeError("GLM-TTS failed to extract prompt mel features from ref_audio.")
        embedding = extract_spk_embedding(wav, int(sr or 24000), self.campplus_session)
        if embedding is None:
            raise RuntimeError("GLM-TTS failed to extract CampPlus speaker embedding from ref_audio.")

        return BatchFeature(
            {
                "input_ids": input_ids,
                "input_len": torch.tensor([int(input_ids.shape[1])], dtype=torch.long),
                "prompt_speech_token": prompt_speech_token_tensor,
                "prompt_speech_token_len": [torch.tensor([len(prompt_speech_token)], dtype=torch.long)],
                "glm_tts_prompt_text_token_len": [torch.tensor([int(prompt_text_ids.shape[1])], dtype=torch.long)],
                "prompt_feat": prompt_feat.detach().to("cpu").unsqueeze(0).contiguous(),
                "embedding": torch.tensor([embedding], dtype=torch.float32),
                "glm_tts_text_token_len": [torch.tensor([int(text_ids.shape[1])], dtype=torch.long)],
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            key: MultiModalFieldConfig.batched("audio")
            for key in (
                "prompt_speech_token",
                "prompt_speech_token_len",
                "glm_tts_prompt_text_token_len",
                "prompt_feat",
                "embedding",
                "glm_tts_text_token_len",
            )
            if key in hf_inputs
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _cached_apply_hf_processor(self, inputs: ProcessorInputs, timing_ctx: Any):
        # GLM-TTS builds the actual AR prompt from both the request text and
        # mm_processor_kwargs["prompt_text"]. The base cache path separates text
        # processing from audio processing, which drops the reference transcript
        # and BOA token from the final prompt.
        return self._apply_hf_processor(inputs, timing_ctx)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def insertion_end(item_idx: int) -> list[int]:
            audio_items = out_mm_kwargs["audio"]
            item = audio_items[item_idx] if item_idx < len(audio_items) else audio_items[0]
            token_len = item["prompt_speech_token_len"].data[0].item()
            return [1] * int(token_len)

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.end(),
                insertion=insertion_end,
            )
        ]


class GLMTTSDummyInputsBuilder(BaseDummyInputsBuilder[GLMTTSMultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "This is a test of the GLM-TTS voice cloning system."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = max(1, int(mm_counts.get("audio") or 1))
        prompt_sample_rate = 24000
        target_audio_length = 3 * prompt_sample_rate
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": (
                self._get_dummy_audios(
                    length=target_audio_length,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                )[0],
                prompt_sample_rate,
            ),
        }

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {"prompt_text": "This is the reference voice."}
        return inputs


@MULTIMODAL_REGISTRY.register_processor(
    GLMTTSMultiModalProcessor,
    info=GLMTTSMultiModalProcessingInfo,
    dummy_inputs=GLMTTSDummyInputsBuilder,
)
class GLMTTSForConditionalGeneration(nn.Module, SupportsMultiModal):
    """vLLM model for GLM-TTS.

    Handles both stages via model_stage branching:
      - ``glm_tts`` (Stage 0): Text → Speech tokens (LLM AR, Llama backbone).
      - ``glm_tts_dit`` (Stage 1): Speech tokens → Audio (DiT flow-matching + vocoder).

    Attributes:
        have_multimodal_outputs: Signals scheduler to collect multimodal outputs.
        has_preprocess: Model has preprocess hook for input preparation (stage 0 only).
        has_postprocess: Model has postprocess hook for hidden state caching (stage 0 only).
    """

    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = True
    prefer_model_sampler = True
    _sampling_eps = 1e-5
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llama_embedding.": "model.embed_tokens.",
            "llama.model.": "model.",
            "llama.": "model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.model_dir = self.model_path
        self.model_stage = getattr(vllm_config.model_config, "model_stage", "glm_tts")

        # Load configuration
        config: GLMTTSConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config

        # ---- Stage 1 (DiT): delegate to GLMTTSDiTForGeneration ----
        if self.model_stage == "glm_tts_dit":
            from .glm_tts_dit_wrapper import GLMTTSDiTForGeneration

            self._dit_gen = GLMTTSDiTForGeneration(vllm_config=vllm_config, prefix=prefix)
            # Expose the DiT as self.model for weight loading routing
            self.model = self._dit_gen
            self.have_multimodal_outputs = True
            self.enable_update_additional_information = True
            # DiT stage does not use preprocess/postprocess/sample
            self.has_preprocess = False
            self.has_postprocess = False
            # DiT loads all weights internally from flow/flow.pt etc.
            # Point DefaultModelLoader to flow/flow.pt (a .pt file) so it can
            # find and load it without triggering the vLLM#39699 bug where
            # subdirectory safetensors patterns cause pt_weights_iterator
            # to be used on .safetensors files.  load_weights() ignores the
            # yielded weights and loads flow/flow.pt itself.
            self.allow_patterns_overrides = ["flow/flow.pt"]
            self.fall_back_to_pt_during_load = False
            return
        self._sample_method = str(getattr(config, "sample_method", "ras")).lower()
        if self._sample_method not in {"ras", "topk"}:
            raise ValueError(f"Unsupported GLM-TTS sample_method={self._sample_method!r}; expected 'ras' or 'topk'.")

        # Resolve repo root to local path (CosyVoice3 pattern).
        # Model weights are in llm/ subdirectory; tokenizer and other resources
        # are siblings of llm/ under the repo root.
        self.model_dir = resolve_glm_tts_model_dir(
            self.model_dir,
            tokenizer_path=getattr(vllm_config.model_config, "tokenizer", None),
        )

        # Stage 0 weights live under llm/model-*.safetensors and are loaded
        # explicitly by _iter_llm_safetensors() during load_weights().
        #
        # We still point DefaultModelLoader at flow/flow.pt as a bootstrap
        # sentinel because the current upstream loader path does not reliably
        # target subdirectory safetensors patterns here without hitting the
        # vLLM#39699 iterator bug. Keep the real stage-0 source of truth in
        # _iter_llm_safetensors(); this override is only to get through the
        # generic loader bootstrap step.
        self.allow_patterns_overrides = ["flow/flow.pt"]
        self.fall_back_to_pt_during_load = False

        # Load tokenizer for special token ID resolution.
        # Prefer the tokenizer path auto-detected by arg_utils
        # (_TOKENIZER_SUBFOLDER_MAP).  Fall back to vq32k-phoneme-tokenizer/
        # under model_dir.
        tokenizer_path = getattr(vllm_config.model_config, "tokenizer", None)
        if tokenizer_path and os.path.isdir(tokenizer_path):
            pass
        else:
            tokenizer_path = os.path.join(self.model_dir, _GLM_TTS_TOKENIZER_SUBDIR)
            if not os.path.exists(tokenizer_path):
                tokenizer_path = self.model_dir
        trust_remote_code = bool(getattr(vllm_config.model_config, "trust_remote_code", False))
        self._tokenizer = load_glm_tts_tokenizer(
            tokenizer_path,
            model_name_or_path=self.model_path,
            trust_remote_code=trust_remote_code,
        )
        special_ids = self._get_special_token_ids()
        self._ats = special_ids["ats"]
        self._ate = special_ids["ate"]
        self._boa = special_ids["boa"]
        self._eoa = special_ids["eoa"]
        self._pad = special_ids["pad"]
        self._bos = self._tokenizer.bos_token_id or self._pad

        logger.info(
            "GLM-TTS token IDs: ATS=%d, ATE=%d, BOA=%d, EOA=%d, PAD=%d, vocab_size=%d",
            self._ats,
            self._ate,
            self._boa,
            self._eoa,
            self._pad,
            config.vocab_size,
        )

        # Validate special token sanity with runtime exceptions (asserts can be
        # stripped under python -O).
        if self._ate - self._ats != 32767:
            raise ValueError(f"Audio token range should be 32768, got {self._ate - self._ats + 1}")
        if self._ats >= self._ate:
            raise ValueError(f"ATS={self._ats} should be < ATE={self._ate}")
        if self._boa >= self._ats:
            raise ValueError(f"BOA={self._boa} should be < ATS={self._ats} (BOA is text token)")

        # Validate vocab_size covers all special tokens
        max_token = max(self._ats, self._ate, self._boa, self._eoa, self._pad)
        if max_token >= config.vocab_size:
            raise ValueError(
                f"vocab_size ({config.vocab_size}) must be > max token ID ({max_token}). "
                f"Check model's config.json has correct vocab_size."
            )

        # Update config with dynamic token IDs so vLLM uses correct eos_token
        # This enables proper stop detection when EOA is sampled
        config.eos_token_id = self._eoa
        config.eoa_token_id = self._eoa
        config.audio_token_start = self._ats
        config.audio_token_end = self._ate
        config.boa_token_id = self._boa
        config.pad_token_id = self._pad
        config.bos_token_id = self._bos

        # Model flags for AR scheduler
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.gpu_resident_buffer_keys: set[str] = {"last_hidden"}

        self._text_frontend: GLMTTSTextFrontend | None = None
        self.model = LlamaModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # LM head for speech token prediction
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Runtime validation: dynamic EOA must match the hardcoded pipeline
        # stop_token_ids constant (59253).  A mismatch means the upstream
        # tokenizer vocabulary has changed and pipeline.py needs updating.
        _PIPELINE_EOA = 59253
        if self._eoa != _PIPELINE_EOA:
            logger.warning(
                "GLM-TTS EOA token mismatch: tokenizer resolved %d but "
                "pipeline.py hardcodes stop_token_ids=[%d]. Update "
                "pipeline.py to match the current checkpoint.",
                self._eoa,
                _PIPELINE_EOA,
            )

        # RAS sampling config — stored as attributes for sample()
        self._ras_win_size = int(getattr(config, "ras_win_size", 10))
        self._ras_tau_r = float(getattr(config, "ras_tau_r", 0.1))
        self._ras_top_p = float(getattr(config, "ras_top_p", 0.8))
        self._ras_top_k = int(getattr(config, "ras_top_k", 25))

        # Thread safety for lazy text frontend loading
        self._text_frontend_lock = threading.Lock()

    def _get_special_token_ids(self) -> dict[str, int]:
        """Get special token IDs from tokenizer dynamically.

        Based on GLM-TTS original: glmtts_inference.py:72-103
        """
        return get_glm_tts_special_token_ids(self._tokenizer)

    def _get_tokenizer(self):
        """Return cached tokenizer (loaded in __init__)."""
        return self._tokenizer

    def _model_dtype(self) -> torch.dtype:
        """Return the active parameter dtype for locally-created embeddings."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            dtype = getattr(self.vllm_config.model_config, "dtype", None)
            if isinstance(dtype, torch.dtype):
                return dtype
            if isinstance(dtype, str):
                resolved = {
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float16": torch.float16,
                    "half": torch.float16,
                    "fp16": torch.float16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }.get(dtype.lower())
                if resolved is not None:
                    return resolved
            return torch.get_default_dtype()

    def _set_generation_bounds_from_text_len(self, text_token_len_raw: Any) -> int | None:
        """Initialize official GLM-TTS AR min/max bounds from target text tokens."""
        text_token_len = _glm_tts_int_value(text_token_len_raw)
        if text_token_len is None:
            return None
        if text_token_len <= 0:
            return None

        min_ratio = float(getattr(self.config, "min_token_text_ratio", 2.0))
        self._glm_tts_min_audio_tokens = max(1, int(text_token_len * min_ratio))
        max_ratio = float(getattr(self.config, "max_token_text_ratio", 20.0))
        self._glm_tts_max_audio_tokens = max(
            self._glm_tts_min_audio_tokens,
            int(text_token_len * max_ratio),
        )
        self._glm_tts_bounds_text_token_len = text_token_len
        return text_token_len

    def _resolve_prefill_target_text_len(
        self,
        span_len: int,
        info_dict: Mapping[str, Any],
        recovered_conditioning: Mapping[str, Any],
    ) -> int | None:
        """Resolve the target-text token count before the first AR sample.

        The processor knows the authoritative target text length, but vLLM's
        multimodal batching can make scalar fields arrive late or with an
        unexpected wrapper.  Prefer lengths inferred from the actual prefill
        layout, then fall back to the explicit scalar.  Never silently use a
        prompt-text length as the target length.
        """

        def field(name: str) -> Any:
            value = info_dict.get(name)
            return recovered_conditioning.get(name) if value is None else value

        provided_text_len = _glm_tts_int_value(field("glm_tts_text_token_len"))
        prompt_text_len = _glm_tts_int_value(field("glm_tts_prompt_text_token_len"))
        prompt_speech_len = _glm_tts_int_value(field("prompt_speech_token_len"))
        input_len = _glm_tts_int_value(info_dict.get("input_len"))

        inferred_text_len: int | None = None
        if input_len is not None and prompt_text_len is not None:
            inferred_text_len = input_len - prompt_text_len - 1  # strip prompt text and BOA.
        elif prompt_text_len is not None and prompt_speech_len is not None:
            inferred_text_len = span_len - prompt_text_len - prompt_speech_len - 1

        if inferred_text_len is not None and inferred_text_len > 0:
            if provided_text_len is not None and provided_text_len != inferred_text_len:
                logger.warning(
                    "GLM-TTS target text token length mismatch: processor=%d inferred=%d; using inferred length.",
                    provided_text_len,
                    inferred_text_len,
                )
            return inferred_text_len

        return provided_text_len if provided_text_len is not None and provided_text_len > 0 else None

    def _recover_prefill_conditioning_from_mm_features(self, info_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Recover request-local conditioning fields before the first postprocess().

        Initial prefill reaches ``preprocess()`` before ``postprocess()`` has
        mirrored multimodal fields into ``model_intermediate_buffer``.  Pull the
        authoritative values from ``mm_features`` so generation bounds and
        diagnostics are available from the first AR forward.
        """
        recovered: dict[str, Any] = {}
        mm_features = info_dict.get("mm_features")
        if not mm_features:
            return recovered
        try:
            feature_kwargs = MultiModalFeatureSpec.gather_kwargs(
                mm_features,
                {"glm_tts_text_token_len", "glm_tts_prompt_text_token_len", "prompt_speech_token_len"},
            )
        except Exception:
            logger.debug("GLM-TTS failed to gather conditioning kwargs from mm_features", exc_info=True)
            return recovered

        for key in ("glm_tts_text_token_len", "glm_tts_prompt_text_token_len", "prompt_speech_token_len"):
            value = feature_kwargs.get(key)
            if value:
                recovered[key] = value
        return recovered

    def _get_text_frontend(self) -> GLMTTSTextFrontend:
        if self._text_frontend is None:
            with self._text_frontend_lock:
                if self._text_frontend is None:
                    self._text_frontend = GLMTTSTextFrontend()
        return self._text_frontend

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any | None = None,
        is_multimodal: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.embed_input_ids(input_ids, **kwargs)
        embed_tokens = self.model.embed_tokens(input_ids)
        if multimodal_embeddings is None or is_multimodal is None:
            return embed_tokens

        mask = torch.as_tensor(is_multimodal, device=embed_tokens.device, dtype=torch.bool).reshape(-1)
        if not bool(mask.any()):
            return embed_tokens

        if isinstance(multimodal_embeddings, (list, tuple)):
            tensors = [
                torch.as_tensor(item, device=embed_tokens.device, dtype=embed_tokens.dtype)
                for item in multimodal_embeddings
                if item is not None
            ]
            if not tensors:
                return embed_tokens
            mm_embeds = torch.cat([item.reshape(-1, item.shape[-1]) for item in tensors], dim=0)
        else:
            mm_embeds = multimodal_embeddings
        if mm_embeds is None:
            return embed_tokens
        mm_embeds = torch.as_tensor(mm_embeds, device=embed_tokens.device, dtype=embed_tokens.dtype)
        if mm_embeds.ndim == 3 and int(mm_embeds.shape[0]) == 1:
            mm_embeds = mm_embeds.squeeze(0)
        if mm_embeds.ndim != 2:
            shape = tuple(mm_embeds.shape)
            raise ValueError(f"GLM-TTS multimodal embeddings should be 2D, got shape={shape}")

        flat_tokens = embed_tokens.reshape(-1, embed_tokens.shape[-1])
        if int(flat_tokens.shape[0]) != int(mask.numel()):
            raise ValueError(
                "GLM-TTS multimodal mask/token length mismatch: "
                f"mask={int(mask.numel())}, tokens={int(flat_tokens.shape[0])}"
            )
        mm_len = int(mask.sum().item())
        if int(mm_embeds.shape[0]) < mm_len:
            raise ValueError(
                "GLM-TTS multimodal embedding length mismatch: "
                f"embeddings={int(mm_embeds.shape[0])}, placeholders={mm_len}"
            )

        flat_tokens = flat_tokens.clone()
        flat_tokens[mask] = mm_embeds[:mm_len]
        return flat_tokens.reshape_as(embed_tokens)

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor] | None:
        if self.model_stage != "glm_tts":
            return None
        prompt_speech_token = kwargs.get("prompt_speech_token")
        if prompt_speech_token is None:
            return None

        def embed_one(value: Any) -> torch.Tensor | None:
            if value is None:
                return None
            speech_token = torch.as_tensor(value, device=next(self.model.parameters()).device)
            if speech_token.numel() == 0:
                return None
            speech_token = speech_token.to(dtype=torch.long)
            if speech_token.ndim == 0:
                speech_token = speech_token.reshape(1, 1)
            elif speech_token.ndim == 1:
                speech_token = speech_token.unsqueeze(0)
            elif speech_token.ndim > 2:
                speech_token = speech_token.reshape(int(speech_token.shape[0]), -1)
            if int(speech_token.min().item()) >= self._ats:
                speech_ids = speech_token
            else:
                speech_ids = speech_token + self._ats
            speech_embeds = self.model.embed_tokens(speech_ids)
            return speech_embeds.reshape(-1, speech_embeds.shape[-1])

        def is_flat_token_sequence(value: list[Any] | tuple[Any, ...]) -> bool:
            if not value:
                return False
            try:
                return all(torch.as_tensor(item).ndim == 0 for item in value)
            except (TypeError, ValueError):
                return False

        if isinstance(prompt_speech_token, (list, tuple)):
            if is_flat_token_sequence(prompt_speech_token):
                one_embed = embed_one(prompt_speech_token)
                return [one_embed] if one_embed is not None else None
            embeds = [embed_one(item) for item in prompt_speech_token]
            embeds = [item for item in embeds if item is not None]
            if not embeds:
                return None
            return embeds

        one_embed = embed_one(prompt_speech_token)
        return [one_embed] if one_embed is not None else None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.forward(input_ids, positions, **kwargs)
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.compute_logits(hidden_states, sampling_metadata)
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None

        return logits

    def _apply_glm_tts_sampling_guard(
        self,
        row_logits: torch.Tensor,
        decoded_tokens: Sequence[int],
    ) -> tuple[torch.Tensor, int, bool, bool]:
        """Apply final GLM-TTS EOA guards.

        The official loop masks EOA immediately before sampling.  Keep that
        ordering here so generic vLLM logits processors cannot undo the
        min/max speech-token bounds.  Do not restrict the rest of the vocab:
        GLM-TTS samples from the full AR distribution and only warns if a
        sampled token falls outside the speech-token range.  The decoded length
        is derived from the request's sampled audio tokens instead of a
        model-global step counter.
        """
        guarded = row_logits.clone()

        audio_len = sum(1 for token in decoded_tokens if self._ats <= int(token) <= self._ate)
        min_tokens = int(getattr(self, "_glm_tts_min_audio_tokens", 0) or 0)
        max_tokens = int(getattr(self, "_glm_tts_max_audio_tokens", 0) or 0)
        eoa_masked = min_tokens > 0 and audio_len < min_tokens
        if eoa_masked:
            guarded[self._eoa] = float("-inf")
        eoa_forced = max_tokens > 0 and audio_len >= max_tokens
        if eoa_forced:
            guarded[:] = float("-inf")
            guarded[self._eoa] = 0.0
        return guarded, audio_len, eoa_masked, eoa_forced

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
    ) -> Any:
        """Custom sampler: RAS (repetition-aware sampling) per CosyVoice3 pattern."""
        if logits is None or logits.numel() == 0:
            return None
        if self.model_stage != "glm_tts":
            return None

        sampler = getattr(self, "_talker_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._talker_sampler = sampler

        if self._sample_method != "ras":
            guarded_logits = logits.to(torch.float32).clone()
            output_token_ids = getattr(sampling_metadata, "output_token_ids", [])
            for req_idx in range(int(guarded_logits.shape[0])):
                decoded_tokens = output_token_ids[req_idx] if req_idx < len(output_token_ids) else []
                guarded_logits[req_idx], _, _, _ = self._apply_glm_tts_sampling_guard(
                    guarded_logits[req_idx],
                    decoded_tokens,
                )
            return sampler(logits=guarded_logits, sampling_metadata=sampling_metadata)

        logits = logits.to(torch.float32)
        try:
            from dataclasses import replace

            sampling_for_processors = replace(sampling_metadata, no_penalties=True)
        except Exception:
            sampling_for_processors = sampling_metadata
        logits = sampler.apply_logits_processors(logits, sampling_for_processors, predict_bonus_token=False)

        ws = getattr(self, "_ras_win_size", 10)
        tr = getattr(self, "_ras_tau_r", 0.1)
        top_p = getattr(self, "_ras_top_p", 0.8)
        top_k = getattr(self, "_ras_top_k", 25)

        sampled_ids: list[int] = []
        for req_idx in range(int(logits.shape[0])):
            row_logits = logits[req_idx]
            temperature = float(req_float(sampling_metadata.temperature, req_idx, 1.0))
            decoded_tokens = (
                sampling_metadata.output_token_ids[req_idx] if req_idx < len(sampling_metadata.output_token_ids) else []
            )
            row_logits, audio_len, eoa_masked, eoa_forced = self._apply_glm_tts_sampling_guard(
                row_logits,
                decoded_tokens,
            )
            if temperature < self._sampling_eps:
                sampled_id = int(torch.argmax(row_logits).item())
                sampled_ids.append(sampled_id)
                continue
            weighted_scores = torch.log_softmax(row_logits / max(temperature, self._sampling_eps), dim=0)
            generator = sampling_metadata.generators.get(req_idx)
            sampled_id = _ras_sample_one(
                weighted_scores,
                decoded_tokens,
                top_p=top_p,
                top_k=top_k,
                win_size=ws,
                tau_r=tr,
                generator=generator,
            )
            sampled_ids.append(sampled_id)

        sampled = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32)
        from vllm.v1.outputs import SamplerOutput

        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        """Package hidden states, speech tokens, and voice clone data into OmniOutput.

        Streaming contract: **delta**.  Each decode step emits exactly one
        speech token (or a prefill placeholder).  The engine's output
        processor concatenates per-step deltas into the final tensor.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []
        if isinstance(info_dicts, dict):
            info_dicts = [info_dicts]

        speech_tokens_list: list[torch.Tensor] = []
        multimodal_extras: dict[str, Any] = {}
        mm_conditioning_keys = (
            "prompt_speech_token",
            "prompt_speech_token_len",
            "glm_tts_prompt_text_token_len",
            "prompt_feat",
            "embedding",
            "glm_tts_text_token_len",
        )
        info_for_flags = next((info for info in info_dicts if isinstance(info, dict)), None)
        if info_for_flags is None or not info_for_flags.get("_voice_clone_emitted"):
            copied_from_kwargs = False
            for key in mm_conditioning_keys:
                val = kwargs.get(key)
                if val is not None:
                    multimodal_extras[key] = val
                    copied_from_kwargs = True
            if copied_from_kwargs and info_for_flags is not None:
                info_for_flags["_voice_clone_emitted"] = True

        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            tokens = info.get("speech_tokens")
            if isinstance(tokens, torch.Tensor) and tokens.numel() > 0:
                speech_tokens_list.append(tokens)
            # Propagate voice cloning features from preprocess info_update.
            # IMPORTANT: Only emit once — the output processor accumulates
            # every decode step and concatenates tensors, which would corrupt
            # constant data (e.g. embedding [192] → [N*192]).
            if not info.get("_voice_clone_emitted"):
                for key in mm_conditioning_keys:
                    val = info.get(key)
                    if val is not None and key not in multimodal_extras:
                        multimodal_extras[key] = val
                if any(info.get(k) is not None for k in mm_conditioning_keys):
                    info["_voice_clone_emitted"] = True

        if not speech_tokens_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs=multimodal_extras)

        speech_tokens = torch.cat(speech_tokens_list, dim=0)
        span_len = int(speech_tokens.shape[0])
        hidden = hidden[:span_len]
        multimodal_extras["speech_tokens"] = speech_tokens
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_extras,
        )

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare inputs for GLM-TTS AR model.

        GLM-TTS only supports the multimodal processor path:
        text prompt + ``multi_modal_data["audio"]`` + ``mm_processor_kwargs["prompt_text"]``.
        Legacy placeholder prompts via ``additional_information`` are rejected.
        """
        if info_dict.get("additional_information") is not None:
            raise ValueError(
                "GLM-TTS no longer accepts legacy additional_information prompts; "
                "use prompt + multi_modal_data['audio'] + mm_processor_kwargs['prompt_text']."
            )

        span_len = int(input_ids.shape[0])
        logger.debug("preprocess: span_len=%d, input_ids.shape=%s", span_len, input_ids.shape)
        if span_len <= 0:
            return input_ids, input_embeds if input_embeds is not None else self.embed_input_ids(input_ids), {}

        device = input_ids.device

        if isinstance(info_dict.get("text"), list) and info_dict["text"]:
            raise ValueError(
                "GLM-TTS no longer accepts legacy text/additional_information prefill payloads; "
                "use the multimodal processor path."
            )

        mm_prefill_done = bool(info_dict.get("glm_tts_mm_prefill_done"))
        recovered_conditioning = self._recover_prefill_conditioning_from_mm_features(info_dict)
        sampled_token = int(input_ids[0].item()) if span_len == 1 else None
        one_token_prefill_tail = (
            input_embeds is not None
            and span_len == 1
            and mm_prefill_done
            and sampled_token is not None
            and sampled_token != self._eoa
            and not (self._ats <= sampled_token <= self._ate)
        )
        is_prefill_span = span_len > 1 or not mm_prefill_done or one_token_prefill_tail
        if input_embeds is None and is_prefill_span:
            raise ValueError("Missing GLM-TTS multimodal input embeddings.")
        if input_embeds is not None and is_prefill_span:
            input_ids_out = input_ids.clone()
            input_ids_out[:] = self._pad
            info_update: dict[str, Any] = {
                "glm_tts_mm_prefill_done": True,
                "speech_tokens": torch.full((span_len, 1), -1, device=device, dtype=torch.long),
            }
            text_token_len = self._resolve_prefill_target_text_len(span_len, info_dict, recovered_conditioning)
            if text_token_len is None:
                raise RuntimeError(
                    "GLM-TTS target text token length is missing before AR prefill. "
                    "Cannot apply the official min/max EOA guard safely."
                )
            self._set_generation_bounds_from_text_len(text_token_len)
            info_update["glm_tts_text_token_len"] = torch.tensor([text_token_len], device=device, dtype=torch.long)
            prompt_text_len = info_dict.get("glm_tts_prompt_text_token_len")
            if prompt_text_len is None:
                prompt_text_len = recovered_conditioning.get("glm_tts_prompt_text_token_len")
            if prompt_text_len is not None:
                info_update["glm_tts_prompt_text_token_len"] = prompt_text_len
            prompt_speech_len = info_dict.get("prompt_speech_token_len")
            if prompt_speech_len is None:
                prompt_speech_len = recovered_conditioning.get("prompt_speech_token_len")
            if prompt_speech_len is not None:
                info_update["prompt_speech_token_len"] = prompt_speech_len
            return input_ids_out, input_embeds.to(dtype=self._model_dtype()), info_update

        # Decode: span_len == 1
        # Standard autoregressive decode - use input_ids directly
        if input_embeds is not None and int(input_embeds.shape[0]) == 1:
            inputs_embeds_out = input_embeds.reshape(1, -1).to(dtype=self._model_dtype())
        else:
            inputs_embeds_out = self.embed_input_ids(input_ids.reshape(1, 1).to(torch.long))
            inputs_embeds_out = inputs_embeds_out.reshape(1, -1).to(dtype=self._model_dtype())

        # Convert sampled token to speech token (relative to ATS)
        # -1 = invalid/EOA, valid range = [0, ATE-ATS]
        sampled_token = int(input_ids[0].item())
        if self._ats <= sampled_token <= self._ate:
            speech_token = sampled_token - self._ats
            if not (0 <= speech_token <= 32767):
                raise ValueError(f"speech_token={speech_token} out of range [0, 32767]")
        else:
            # EOA or other non-audio token → mark as invalid (-1)
            speech_token = -1
            logger.debug("GLM-TTS decode: non-audio token %d (EOA=%d)", sampled_token, self._eoa)
        speech_tokens = torch.tensor([[speech_token]], device=device, dtype=torch.long)

        info_update = {"speech_tokens": speech_tokens}
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        """Cache last hidden state for next decode step."""
        if hidden_states.numel() == 0:
            return {}
        last = hidden_states[-1, :].detach()
        update: dict[str, Any] = {"last_hidden": last}

        multimodal_outputs = kwargs.get("multimodal_outputs")
        if isinstance(multimodal_outputs, dict):
            copied_conditioning = False
            for key in ("prompt_speech_token", "prompt_speech_token_len", "prompt_feat", "embedding"):
                val = multimodal_outputs.get(key)
                if val is not None:
                    update[key] = _first_glm_tts_value(val)
                    copied_conditioning = True

            prompt_text_len = multimodal_outputs.get("glm_tts_prompt_text_token_len")
            if prompt_text_len is not None:
                update["glm_tts_prompt_text_token_len"] = _first_glm_tts_value(prompt_text_len)

            text_token_len = _glm_tts_int_value(multimodal_outputs.get("glm_tts_text_token_len"))
            if text_token_len is not None:
                existing_text_len = getattr(self, "_glm_tts_bounds_text_token_len", None)
                if existing_text_len is None:
                    self._set_generation_bounds_from_text_len(text_token_len)
                    update["glm_tts_text_token_len"] = text_token_len
                elif int(existing_text_len) != int(text_token_len):
                    logger.warning(
                        "Ignoring late GLM-TTS text length update: existing=%d late=%d",
                        int(existing_text_len),
                        int(text_token_len),
                    )
                    update["glm_tts_text_token_len"] = int(existing_text_len)
                else:
                    update["glm_tts_text_token_len"] = text_token_len

            if copied_conditioning:
                update["_voice_clone_emitted"] = True

        return update

    def _iter_llm_safetensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        """Yield (name, tensor) pairs from llm/model-*.safetensors."""
        import glob as glob_module

        from safetensors.torch import load_file

        llm_dir = os.path.join(self.model_dir, "llm")
        if os.path.isdir(llm_dir):
            sf_files = sorted(glob_module.glob(os.path.join(llm_dir, "model-*.safetensors")))
            if sf_files:
                for sf_path in sf_files:
                    yield from load_file(sf_path, device="cpu").items()
                return

        from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

        model_root = download_weights_from_hf_specific(
            self.model_path,
            self.vllm_config.load_config.download_dir,
            allow_patterns=["llm/model-*.safetensors"],
        )
        llm_dir = os.path.join(model_root, "llm")
        sf_files = sorted(glob_module.glob(os.path.join(llm_dir, "model-*.safetensors")))
        if not sf_files:
            raise RuntimeError(f"No LLM safetensors found under {model_root}. Expected llm/model-*.safetensors.")
        for sf_path in sf_files:
            yield from load_file(sf_path, device="cpu").items()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.

        Stage 0 (glm_tts): HuggingFace Llama-format checkpoint from llm/ subdir.
        Stage 1 (glm_tts_dit): DiT flow.pt + vocoder.
        """
        if self.model_stage == "glm_tts_dit":
            return self._dit_gen.load_weights(weights)

        def _glm_tts_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, loaded_weight in self._iter_llm_safetensors():
                if "rotary_emb.inv_freq" in name:
                    continue
                if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                    continue
                yield name, loaded_weight

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(_glm_tts_weights(), mapper=self.hf_to_vllm_mapper)

        params_dict = dict(self.named_parameters())
        # Handle tie_word_embeddings: copy embed_tokens to lm_head if not loaded.
        lm_head_key = "lm_head.weight"
        embed_key = "model.embed_tokens.weight"
        if lm_head_key not in loaded_params and embed_key in loaded_params:
            if lm_head_key in params_dict and embed_key in params_dict:
                lm_head_param = params_dict[lm_head_key]
                embed_param = params_dict[embed_key]
                weight_loader = getattr(lm_head_param, "weight_loader", default_weight_loader)
                weight_loader(lm_head_param, embed_param.data)
                loaded_params.add(lm_head_key)
                logger.info("Tied lm_head.weight to embed_tokens.weight")

        logger.info("Loaded %d weights for GLMTTSForConditionalGeneration", len(loaded_params))
        return loaded_params
