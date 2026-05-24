# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External model loading, audio I/O, and emotion conditioning for IndexTTS2."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils.hub import cached_file
from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy external model singletons
# ---------------------------------------------------------------------------

_wav2vec2_model = None
_wav2vec2_processor = None
_semantic_codec = None
_campplus_model = None
_qwen_emotion_model = None
_qwen_emotion_tokenizer = None


def resolve_model_file(model_path: str, filename: str) -> str | None:
    """Resolve an IndexTTS2 asset from a local model dir or HF repo id."""
    local_path = os.path.join(model_path, filename)
    if os.path.isfile(local_path):
        return local_path
    try:
        return cached_file(model_path, filename)
    except Exception as e:
        logger.warning("Could not resolve IndexTTS2 asset %s from %s: %s", filename, model_path, e)
        return None


def resolve_model_dir(model_path: str, dirname: str) -> str | None:
    """Resolve a local-only subdirectory, returning None for remote repos."""
    local_path = os.path.join(model_path, dirname)
    return local_path if os.path.isdir(local_path) else None


def load_wav2vec2(model_path: str, device: torch.device):
    global _wav2vec2_model, _wav2vec2_processor
    if _wav2vec2_model is not None:
        return _wav2vec2_model, _wav2vec2_processor
    from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    w2v_path = resolve_model_dir(model_path, "wav2vec2bert")
    if w2v_path is None:
        w2v_path = "facebook/w2v-bert-2.0"
    _wav2vec2_processor = AutoFeatureExtractor.from_pretrained(w2v_path)
    _wav2vec2_model = Wav2Vec2BertModel.from_pretrained(w2v_path)
    _wav2vec2_model = _wav2vec2_model.to(device=device, dtype=torch.float32).eval()
    for p in _wav2vec2_model.parameters():
        p.requires_grad_(False)
    return _wav2vec2_model, _wav2vec2_processor


def load_semantic_codec(model_path: str, config: dict, device: torch.device):
    global _semantic_codec
    if _semantic_codec is not None:
        return _semantic_codec
    from .utils.maskgct.repcodec_model import RepCodec

    codec = RepCodec(
        codebook_size=config.get("codebook_size", 8192),
        hidden_size=config.get("hidden_size", 1024),
        codebook_dim=config.get("codebook_dim", 8),
    )
    ckpt_path = resolve_model_file(model_path, "semantic_codec.pth")
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        codec.load_state_dict(state, strict=False)
    else:
        import safetensors.torch
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(codec, ckpt_path)
    codec = codec.to(device=device, dtype=torch.float32).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    _semantic_codec = codec
    return _semantic_codec


def load_campplus(model_path: str, device: torch.device):
    global _campplus_model
    if _campplus_model is not None:
        return _campplus_model
    from .utils.campplus.dtdnn import CAMPPlus

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    ckpt_path = resolve_model_file(model_path, "campplus.pth")
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    campplus.load_state_dict(state, strict=False)
    campplus = campplus.to(device=device, dtype=torch.float32).eval()
    for p in campplus.parameters():
        p.requires_grad_(False)
    _campplus_model = campplus
    return _campplus_model


def load_qwen_emotion(model_path: str, device: torch.device, *, trust_remote_code: bool = True):
    global _qwen_emotion_model, _qwen_emotion_tokenizer
    if _qwen_emotion_model is not None:
        return _qwen_emotion_model, _qwen_emotion_tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    qwen_emo_path = resolve_model_dir(model_path, "qwen0.6bemo4-merge")
    if qwen_emo_path is None:
        raise FileNotFoundError(
            f"QwenEmotion model directory 'qwen0.6bemo4-merge' was not found in {model_path}. "
            "It is required when IndexTTS2 use_emo_text=True."
        )
    _qwen_emotion_tokenizer = AutoTokenizer.from_pretrained(
        qwen_emo_path,
        trust_remote_code=trust_remote_code,
    )
    _qwen_emotion_model = AutoModelForCausalLM.from_pretrained(
        qwen_emo_path,
        torch_dtype="float16",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    _qwen_emotion_model.eval()
    for p in _qwen_emotion_model.parameters():
        p.requires_grad_(False)
    return _qwen_emotion_model, _qwen_emotion_tokenizer


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def compute_fbank(wav_16k: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute 80-dim fbank features for CAMPPlus. Input: [T] at 16kHz."""
    try:
        import torchaudio.compliance.kaldi as kaldi

        wav = wav_16k.unsqueeze(0).float().cpu()
        fbank = kaldi.fbank(
            wav,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        return fbank.unsqueeze(0).to(device=device)  # [1, T, 80]
    except (ImportError, RuntimeError, OSError):
        logger.warning("torchaudio not available, returning zero fbank")
        return torch.zeros(1, 100, 80, device=device)


def wav2vec_extract(
    wav_16k: torch.Tensor,
    model: Any,
    processor: Any,
    device: torch.device,
    w2v_stat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract Wav2Vec2-BERT features. Returns [1, T, 1024]."""
    wav_np = wav_16k.cpu().numpy()
    inputs = processor(wav_np, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_features, output_hidden_states=True)
    feat = outputs.hidden_states[17]
    if w2v_stat is not None:
        mean = w2v_stat[0:1, :].to(device=device)
        std = torch.sqrt(w2v_stat[1:2, :].to(device=device))
        feat = (feat - mean) / std
    return feat


def load_reference_audio(
    audio_path: str | tuple | list,
    device: torch.device,
    max_audio_length_seconds: float | None = 15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load reference audio and resample to 16kHz and 22.05kHz.

    Accepts either a file path (str) or a pre-loaded (wav_list, sr) tuple
    from the serving layer.
    """
    if isinstance(audio_path, (list, tuple)) and len(audio_path) == 2:
        wav_data, sr = audio_path
        if isinstance(wav_data, np.ndarray):
            wav = torch.from_numpy(wav_data).float()
        elif isinstance(wav_data, list):
            wav = torch.tensor(wav_data, dtype=torch.float32)
        elif isinstance(wav_data, torch.Tensor):
            wav = wav_data.float()
        else:
            raise TypeError(f"Unsupported audio data type: {type(wav_data)}")
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        wav = _truncate_audio(wav, int(sr), max_audio_length_seconds)
        wav_16k = _resample(wav, sr, 16000)
        wav_22k = _resample(wav, sr, 22050)
        return wav_16k, wav_22k

    try:
        import torchaudio

        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        wav = _truncate_audio(wav, int(sr), max_audio_length_seconds)
        wav_16k = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
        wav_22k = torchaudio.functional.resample(wav, sr, 22050) if sr != 22050 else wav
        return wav_16k, wav_22k
    except (ImportError, RuntimeError, OSError):
        pass

    import soundfile as sf

    audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)
    wav = torch.from_numpy(audio_np).float()
    wav = _truncate_audio(wav, int(sr), max_audio_length_seconds)
    wav_16k = _resample(wav, sr, 16000)
    wav_22k = _resample(wav, sr, 22050)
    return wav_16k, wav_22k


def _truncate_audio(
    wav: torch.Tensor,
    sample_rate: int,
    max_audio_length_seconds: float | None,
) -> torch.Tensor:
    if max_audio_length_seconds is None:
        return wav
    max_audio_samples = int(max_audio_length_seconds * sample_rate)
    if max_audio_samples > 0 and wav.shape[-1] > max_audio_samples:
        return wav[..., :max_audio_samples]
    return wav


def _resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    try:
        import torchaudio

        return torchaudio.functional.resample(wav, orig_sr, target_sr)
    except (ImportError, RuntimeError, OSError):
        return F.interpolate(
            wav.unsqueeze(0).unsqueeze(0), scale_factor=target_sr / orig_sr, mode="linear", align_corners=False
        ).squeeze()
