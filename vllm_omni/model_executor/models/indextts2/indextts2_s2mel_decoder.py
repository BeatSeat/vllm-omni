# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS2 Stage 1: S2Mel decoder + BigVGAN vocoder.

Receives mel_codes + latent from Stage 0 (GPT AR talker), runs flow matching
to synthesize mel spectrogram, then BigVGAN to produce waveform audio.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_indextts2 import IndexTTS2Config
from .preprocess_utils import resolve_model_file
from .s2mel.modules.commons import AttrDict, MyModel

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy loaders for external models
# ---------------------------------------------------------------------------

_bigvgan_models: dict[tuple[str, str], nn.Module] = {}
_semantic_codec_decoder = None


def _patch_bigvgan_compat(cls):
    """Monkey-patch BigVGAN._from_pretrained for huggingface_hub>=1.0 compat."""
    orig = cls._from_pretrained.__func__

    @classmethod
    def _compat(klass, *, proxies=None, resume_download=False, **kw):
        return orig(klass, proxies=proxies, resume_download=resume_download, **kw)

    cls._from_pretrained = _compat


def _load_bigvgan(vocoder_name: str, device: torch.device):
    cache_key = (vocoder_name, str(device))
    if cache_key in _bigvgan_models:
        return _bigvgan_models[cache_key]

    try:
        from .s2mel.modules import bigvgan as bigvgan_mod

        _patch_bigvgan_compat(bigvgan_mod.BigVGAN)
        bigvgan_model = bigvgan_mod.BigVGAN.from_pretrained(vocoder_name)
    except (ImportError, ModuleNotFoundError):
        import bigvgan

        _patch_bigvgan_compat(bigvgan.BigVGAN)
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_name)
    bigvgan_model = bigvgan_model.to(device=device).eval()
    bigvgan_model.remove_weight_norm()
    for p in bigvgan_model.parameters():
        p.requires_grad_(False)
    _bigvgan_models[cache_key] = bigvgan_model
    return bigvgan_model


def _load_semantic_codec_for_vq2emb(model_path: str, config: dict, device: torch.device):
    global _semantic_codec_decoder
    if _semantic_codec_decoder is not None:
        return _semantic_codec_decoder
    from .preprocess_utils import load_semantic_codec

    _semantic_codec_decoder = load_semantic_codec(model_path, config, device)
    return _semantic_codec_decoder


# ---------------------------------------------------------------------------
# Stage 1 Decoder
# ---------------------------------------------------------------------------


class IndexTTS2S2MelDecoder(nn.Module):
    """S2Mel + BigVGAN decoder for IndexTTS2 Stage 1.

    Receives from Stage 0:
      - mel_codes: [T] mel code sequence
      - latent: [T, 1280] hidden states from GPT AR
      - S_ref: speaker semantic embeddings (from Wav2Vec2 → RepCodec)
      - ref_mel: [80, T_ref] reference mel spectrogram
      - style: [192] CAMPPlus style vector
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: IndexTTS2Config = vllm_config.model_config.hf_config  # type: ignore[assignment]

        # --- Flags for vLLM-Omni framework ---
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # --- Build S2Mel model (CFM + LengthRegulator + gpt_layer) ---
        s2mel_cfg = self.config.s2mel
        s2mel_args = AttrDict(s2mel_cfg)
        # Ensure nested dicts are also AttrDicts
        for key in ["DiT", "wavenet", "length_regulator", "style_encoder", "preprocess_params"]:
            if key in s2mel_args and isinstance(s2mel_args[key], dict):
                s2mel_args[key] = AttrDict(s2mel_args[key])

        self.s2mel = MyModel(s2mel_args, use_emovec=False, use_gpt_latent=True)

        # Diffusion config
        self.diffusion_steps = 25
        self.inference_cfg_rate = 0.7
        self.mel_code_to_frame_ratio = 1.72
        self._s2mel_torch_compile_attempted = False

    # ------------------------------------------------------------------
    # vLLM hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OmniOutput:
        """Run S2Mel flow matching + BigVGAN vocoding."""
        model_intermediate_buffer = kwargs.get("model_intermediate_buffer")
        if model_intermediate_buffer is None:
            model_intermediate_buffer = runtime_additional_information
        if runtime_additional_information is not None and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")

        additional_information: dict[str, Any] = {}
        if model_intermediate_buffer and len(model_intermediate_buffer) > 0:
            additional_information = model_intermediate_buffer[0]

        device = input_ids.device
        model_dtype = self.s2mel.models["gpt_layer"][0].weight.dtype

        # Log raw additional_information shapes
        logger.info(
            "[S2Mel forward] model_intermediate_buffer len=%d, keys=%s",
            len(model_intermediate_buffer) if model_intermediate_buffer else 0,
            list(additional_information.keys()) if additional_information else [],
        )
        for k, v in additional_information.items():
            if isinstance(v, torch.Tensor):
                logger.info("[S2Mel forward] %s: shape=%s, dtype=%s", k, v.shape, v.dtype)

        # Extract Stage 0 outputs and cast to model dtype
        latent = self._get_tensor(additional_information, "latent", device, model_dtype)
        mel_codes = self._get_tensor(additional_information, "mel_codes", device)
        code_lens_raw = additional_information.get("code_lens")
        s_ref = self._get_tensor(additional_information, "S_ref", device)
        ref_mel = self._get_tensor(additional_information, "ref_mel", device, model_dtype)
        style = self._get_tensor(additional_information, "style", device, model_dtype)

        if mel_codes is None or latent is None:
            logger.warning("S2Mel decoder received empty mel_codes or latent")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, device=device),
                multimodal_outputs={"audio": torch.zeros(1, device=device), "sr": 22050},
            )

        logger.info(
            "[S2Mel] extracted tensors — mel_codes=%s latent=%s code_lens=%s s_ref=%s ref_mel=%s style=%s",
            mel_codes.shape if mel_codes is not None else None,
            latent.shape if latent is not None else None,
            code_lens_raw,
            s_ref.shape if isinstance(s_ref, torch.Tensor) else None,
            ref_mel.shape if isinstance(ref_mel, torch.Tensor) else None,
            style.shape if isinstance(style, torch.Tensor) else None,
        )

        # Ensure batch dimension
        if mel_codes.ndim == 1:
            mel_codes = mel_codes.unsqueeze(0)
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)

        # Compute code_lens
        if code_lens_raw is not None:
            if isinstance(code_lens_raw, torch.Tensor):
                code_lens = code_lens_raw.to(device=device, dtype=torch.long)
            else:
                code_lens = torch.tensor(code_lens_raw, device=device, dtype=torch.long)
        else:
            # Strip stop token (8193) and compute actual length
            stop_token = self.config.gpt.get("stop_mel_token", 8193)
            code_lens = []
            for i in range(mel_codes.shape[0]):
                stop_mask = (mel_codes[i] == stop_token).nonzero(as_tuple=False)
                if stop_mask.numel() > 0:
                    code_lens.append(stop_mask[0].item())
                else:
                    code_lens.append(mel_codes.shape[1])
            code_lens = torch.tensor(code_lens, device=device, dtype=torch.long)

        # Trim to actual length
        max_len = int(code_lens.max().item())
        mel_codes = mel_codes[:, :max_len]
        latent = latent[:, :max_len, :]
        logger.info(
            "[S2Mel] after trim — mel_codes=%s latent=%s code_lens=%s max_len=%d",
            mel_codes.shape,
            latent.shape,
            code_lens.tolist(),
            max_len,
        )

        # --- S2Mel pipeline ---
        # 1. Project GPT latent: 1280 → 1024
        latent = latent.to(device=device, dtype=model_dtype)
        latent = self.s2mel.forward_gpt(latent).to(dtype=model_dtype)  # [B, T, 1024]
        logger.info("[S2Mel] step1 forward_gpt → latent=%s dtype=%s", latent.shape, latent.dtype)

        # 2. Embed mel codes via semantic codec vq2emb
        semantic_codec = _load_semantic_codec_for_vq2emb(self.model_path, self.config.semantic_codec, device)
        codebook_size = self.config.semantic_codec.get("codebook_size", 8192)
        mel_codes_clamped = mel_codes.clamp(0, codebook_size - 1)
        logger.info(
            "mel_codes debug: shape=%s, min=%d, max=%d",
            mel_codes.shape,
            mel_codes.min().item(),
            mel_codes.max().item(),
        )
        with torch.no_grad():
            S_infer = semantic_codec.quantizer.vq2emb(mel_codes_clamped.unsqueeze(1))  # [B, T, 1024]
        S_infer = S_infer.transpose(1, 2).to(dtype=model_dtype)  # [B, T, 1024]
        S_infer = S_infer + latent
        logger.info("[S2Mel] step2 vq2emb → S_infer=%s dtype=%s", S_infer.shape, S_infer.dtype)

        # 3. Length regulate: codes → mel frames
        target_lengths = (code_lens.float() * self.mel_code_to_frame_ratio).long()
        logger.info("[S2Mel] step3 length_regulator target_lengths=%s", target_lengths.tolist())
        cond = self.s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[
            0
        ]  # [B, T_mel, 512]
        logger.info("[S2Mel] step3 length_regulator → cond=%s", cond.shape)

        # 4. Reference prompt conditioning
        if s_ref is not None and ref_mel is not None:
            if ref_mel.ndim == 2:
                ref_mel = ref_mel.unsqueeze(0)

            if s_ref.ndim == 3:
                # Official infer_v2.py uses the quantized embedding returned by
                # semantic_codec.quantize() directly as S_ref.
                s_ref_emb = s_ref.to(device=device, dtype=model_dtype)
            else:
                # Backward-compatible path for older payloads that carried
                # codebook indices instead of quantized embeddings.
                s_ref_codes = s_ref.long()
                if s_ref_codes.ndim == 1:
                    s_ref_codes = s_ref_codes.unsqueeze(0)  # [1, T]
                codebook_size = self.config.semantic_codec.get("codebook_size", 8192)
                logger.info(
                    "S_ref debug: shape=%s, min=%d, max=%d, codebook_size=%d",
                    s_ref_codes.shape,
                    s_ref_codes.min().item(),
                    s_ref_codes.max().item(),
                    codebook_size,
                )
                s_ref_codes = s_ref_codes.clamp(0, codebook_size - 1)
                with torch.no_grad():
                    s_ref_emb = semantic_codec.quantizer.vq2emb(s_ref_codes.unsqueeze(1))  # [B, 1024, T]
                s_ref_emb = s_ref_emb.transpose(1, 2).to(dtype=model_dtype)  # [B, T, 1024]

            ref_target_lengths = torch.tensor([ref_mel.size(-1)], device=device, dtype=torch.long)
            prompt_condition = self.s2mel.models["length_regulator"](
                s_ref_emb, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            logger.info(
                "[S2Mel] step4 ref conditioning — s_ref_emb=%s prompt_condition=%s cat_condition=%s",
                s_ref_emb.shape,
                prompt_condition.shape,
                cat_condition.shape,
            )
        else:
            cat_condition = cond
            ref_mel = torch.zeros(1, 80, 1, device=device, dtype=model_dtype)

        if style is None:
            style = torch.zeros(1, 192, device=device, dtype=model_dtype)
        if style.ndim == 1:
            style = style.unsqueeze(0)

        logger.info(
            "[S2Mel] step5 CFM input — cat_condition=%s ref_mel=%s style=%s steps=%d cfg_rate=%.2f",
            cat_condition.shape,
            ref_mel.shape,
            style.shape,
            self.diffusion_steps,
            self.inference_cfg_rate,
        )

        # 5. Flow matching inference (25 Euler steps, CFG rate 0.7)
        # CFM ODE solver must run in float32 — fp16 Euler steps diverge to noise
        cfm = self.s2mel.models["cfm"]
        cfm.float()
        cat_condition = cat_condition.float()
        ref_mel = ref_mel.float()
        style = style.float()

        estimator = cfm.estimator
        if estimator.transformer.freqs_cis is None:
            estimator.setup_caches(max_batch_size=2, max_seq_length=16384)
            logger.info("[S2Mel] DiT caches initialized (freqs_cis + causal_mask)")
        self._maybe_enable_s2mel_torch_compile()

        x_lens = torch.tensor([cat_condition.size(1)], device=device, dtype=torch.long)
        try:
            with torch.no_grad():
                mel = cfm.inference(
                    cat_condition,
                    x_lens,
                    ref_mel,
                    style,
                    None,  # f0
                    self.diffusion_steps,
                    inference_cfg_rate=self.inference_cfg_rate,
                )
        except Exception as exc:
            if not getattr(cfm, "_compiled", False) or not hasattr(cfm, "disable_torch_compile"):
                raise
            logger.warning("IndexTTS2 S2Mel compiled inference failed; retrying eager S2Mel: %s", exc)
            cfm.disable_torch_compile()
            with torch.no_grad():
                mel = cfm.inference(
                    cat_condition,
                    x_lens,
                    ref_mel,
                    style,
                    None,  # f0
                    self.diffusion_steps,
                    inference_cfg_rate=self.inference_cfg_rate,
                )
        # Strip reference portion
        mel = mel[:, :, ref_mel.size(-1) :]
        logger.info("[S2Mel] step5 CFM done → mel=%s (stripped ref %d frames)", mel.shape, ref_mel.size(-1))

        # 6. BigVGAN vocoding
        vocoder_cfg = self.config.vocoder
        vocoder_name = vocoder_cfg.get("name", "nvidia/bigvgan_v2_22khz_80band_256x")
        bigvgan = _load_bigvgan(vocoder_name, device)
        with torch.no_grad():
            wav = bigvgan(mel.float()).squeeze()

        # Keep the public vLLM-Omni audio contract as normalized float. This is
        # numerically equivalent to official infer_v2.py before its int16 save
        # step, while avoiding double scaling in OpenAI/soundfile encoders.
        wav = torch.clamp(wav, -1.0, 1.0)
        if wav.ndim == 0:
            wav = wav.unsqueeze(0)

        logger.info(
            "[S2Mel] step6 BigVGAN done → wav=%s min=%.1f max=%.1f sr=22050",
            wav.shape,
            wav.min().item(),
            wav.max().item(),
        )

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": wav.cpu(),
                "sr": torch.tensor(22050, dtype=torch.int32),
            },
        )

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tensor(
        info: dict,
        key: str,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """Extract tensor from additional_information, handling nested dicts."""
        val = info.get(key)
        if val is None:
            for sub_key in ["codes", "hidden_states", "meta"]:
                sub = info.get(sub_key)
                if isinstance(sub, dict) and key in sub:
                    val = sub[key]
                    break
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.to(device=device, dtype=dtype) if dtype else val.to(device=device)
        if isinstance(val, list):
            if val and isinstance(val[0], torch.Tensor):
                return val[0].to(device=device, dtype=dtype) if dtype else val[0].to(device=device)
        return None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from s2mel.pth checkpoint.

        Bypasses vllm's default .pt iterator (same reason as talker: raw
        tensor .pt files in model dir). Loads s2mel.pth directly and
        flattens the nested ``state["net"]`` structure:
          net["cfm"][k]              → s2mel.models.cfm.{k}
          net["length_regulator"][k] → s2mel.models.length_regulator.{k}
          net["gpt_layer"][k]        → s2mel.models.gpt_layer.{k}
        """
        _ = weights

        ckpt_path = resolve_model_file(self.model_path, self.config.s2mel_checkpoint)
        if ckpt_path is None:
            raise FileNotFoundError(f"IndexTTS2 S2Mel checkpoint {self.config.s2mel_checkpoint!r} not found")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        net = state["net"] if "net" in state else state

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        prefix_map = {
            "cfm.": "s2mel.models.cfm.",
            "length_regulator.": "s2mel.models.length_regulator.",
            "gpt_layer.": "s2mel.models.gpt_layer.",
        }

        # Flatten nested ModuleDict state: {module: {param: tensor}} → flat iter
        flat_items: Iterable[tuple[str, torch.Tensor]]
        if net and isinstance(next(iter(net.values())), dict):
            flat_items = (
                (f"{mod}.{param}", tensor) for mod, mod_state in net.items() for param, tensor in mod_state.items()
            )
        else:
            flat_items = net.items()

        for name, loaded_weight in flat_items:
            mapped_name = name
            for old_prefix, new_prefix in prefix_map.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix) :]
                    break

            if mapped_name not in params_dict:
                logger.debug("Skipping unrecognized S2Mel weight: %s → %s", name, mapped_name)
                continue

            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped_name)

        logger.info("Loaded %d weights for IndexTTS2S2MelDecoder", len(loaded_params))
        required_prefixes = [
            "s2mel.models.cfm.",
            "s2mel.models.length_regulator.",
            "s2mel.models.gpt_layer.",
        ]
        missing_prefixes = [
            prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in loaded_params)
        ]
        if missing_prefixes:
            raise RuntimeError(
                "IndexTTS2 S2Mel checkpoint did not load required parameter groups: " + ", ".join(missing_prefixes)
            )
        return loaded_params

    def _maybe_enable_s2mel_torch_compile(self) -> None:
        if self._s2mel_torch_compile_attempted:
            return
        self._s2mel_torch_compile_attempted = True

        if getattr(self.vllm_config.model_config, "enforce_eager", False):
            logger.info("Skipping IndexTTS2 S2Mel torch.compile because enforce_eager=True")
            return

        if not hasattr(self.s2mel, "enable_torch_compile"):
            logger.warning("IndexTTS2 S2Mel does not expose enable_torch_compile")
            return

        try:
            self.s2mel.enable_torch_compile()
        except Exception as exc:
            logger.warning("IndexTTS2 S2Mel torch.compile failed; using eager S2Mel: %s", exc)
        else:
            logger.info("Enabled IndexTTS2 S2Mel torch.compile")
