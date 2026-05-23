# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for IndexTTS2: Talker (GPT AR) → S2Mel decoder."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _strip_stop_token(
    codes: torch.Tensor,
    latent: torch.Tensor,
    stop_mel_token: int = 8193,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strip at the first stop token, matching official IndexTTS2 v2.

    Returns: (codes [B, T'], latent [B, T', D], code_lens [B]).
    """
    if codes.ndim == 1:
        codes = codes.unsqueeze(0)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)

    device = codes.device
    code_lens = []
    codes_out = []
    latent_out = []

    for i in range(codes.shape[0]):
        code = codes[i]
        lat = latent[i]

        # Find stop token
        stop_mask = (code == stop_mel_token).nonzero(as_tuple=False)
        if stop_mask.numel() > 0:
            valid_len = int(stop_mask[0].item())
        else:
            valid_len = int(code.shape[0])
        code = code[:valid_len]
        lat = lat[:valid_len]

        code_lens.append(int(code.shape[0]))
        codes_out.append(code)
        latent_out.append(lat)

    # Pad to max length
    max_len = max(code_lens) if code_lens else 0
    if max_len == 0:
        return (
            torch.zeros(codes.shape[0], 0, dtype=torch.long, device=device),
            torch.zeros(codes.shape[0], 0, latent.shape[-1], device=device, dtype=latent.dtype),
            torch.zeros(codes.shape[0], dtype=torch.long, device=device),
        )

    padded_codes = torch.full((len(codes_out), max_len), stop_mel_token, dtype=torch.long, device=device)
    lat_dtype = latent_out[0].dtype
    padded_latent = torch.zeros(
        len(latent_out),
        max_len,
        latent_out[0].shape[-1],
        device=device,
        dtype=lat_dtype,
    )
    for i, (c, lat) in enumerate(zip(codes_out, latent_out)):
        padded_codes[i, : c.shape[0]] = c
        padded_latent[i, : lat.shape[0]] = lat

    return padded_codes, padded_latent, torch.tensor(code_lens, dtype=torch.long, device=device)


def talker2s2mel(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all Stage 0 output, format for Stage 1 S2Mel decoder."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = source_outputs
    s2mel_inputs: list[OmniTokensPrompt] = []

    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            continue

        output = talker_output.outputs[0]
        mm = output.multimodal_output
        if not isinstance(mm, dict):
            logger.warning("Talker output %d has no multimodal_output dict", i)
            continue

        # Extract mel_codes
        codes_dict = mm.get("codes", {})
        mel_codes = codes_dict.get("mel")
        if not isinstance(mel_codes, torch.Tensor) or mel_codes.numel() == 0:
            logger.warning("Talker output %d has empty mel_codes", i)
            continue
        mel_codes = mel_codes.to(torch.long)

        # Extract latent hidden states
        hs_dict = mm.get("hidden_states", {})
        latent = hs_dict.get("latent")
        if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
            logger.warning("Talker output %d has empty latent", i)
            continue

        # Extract metadata from Stage 0
        meta = mm.get("meta", {})
        s_ref = meta.get("S_ref")
        ref_mel = meta.get("ref_mel")
        style = meta.get("style")

        logger.info(
            "[talker2s2mel] shapes — mel_codes=%s, latent=%s, S_ref=%s, ref_mel=%s, style=%s",
            mel_codes.shape if isinstance(mel_codes, torch.Tensor) else None,
            latent.shape if isinstance(latent, torch.Tensor) else None,
            s_ref.shape if isinstance(s_ref, torch.Tensor) else None,
            ref_mel.shape if isinstance(ref_mel, torch.Tensor) else None,
            style.shape if isinstance(style, torch.Tensor) else None,
        )

        # Official infer_v2.py only strips at the first stop token in this path.
        mel_codes_clean, latent_clean, code_lens = _strip_stop_token(mel_codes, latent)

        logger.info(
            "[talker2s2mel] after stop trim — mel_codes=%s→%s, latent=%s→%s, code_lens=%s",
            mel_codes.shape,
            mel_codes_clean.shape,
            latent.shape,
            latent_clean.shape,
            code_lens.tolist(),
        )

        # Build additional_information for Stage 1
        additional_information = {
            "latent": latent_clean.float().cpu(),
            "mel_codes": mel_codes_clean.cpu(),
            "code_lens": code_lens.cpu(),
        }
        if isinstance(s_ref, torch.Tensor):
            additional_information["S_ref"] = s_ref.float().cpu()
        else:
            logger.warning("[talker2s2mel] S_ref MISSING — Stage 1 will skip ref conditioning")
        if isinstance(ref_mel, torch.Tensor):
            additional_information["ref_mel"] = ref_mel.float().cpu()
        else:
            logger.warning("[talker2s2mel] ref_mel MISSING — Stage 1 will skip ref conditioning")
        if isinstance(style, torch.Tensor):
            additional_information["style"] = style.float().cpu()
        else:
            logger.warning("[talker2s2mel] style MISSING — Stage 1 will use zeros")

        s2mel_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # dummy token for vLLM scheduler
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information if additional_information else None,
            )
        )

    return s2mel_inputs
