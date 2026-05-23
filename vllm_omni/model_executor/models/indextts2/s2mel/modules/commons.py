# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act_part, s_act_part = torch.split(in_act, n_channels_int, dim=1)
    t_act = torch.tanh(t_act_part)
    s_act = torch.sigmoid(s_act_part)
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class MyModel(nn.Module):
    def __init__(self, args, use_emovec=False, use_gpt_latent=False):
        super().__init__()
        from vllm_omni.model_executor.models.indextts2.s2mel.modules.flow_matching import CFM
        from vllm_omni.model_executor.models.indextts2.s2mel.modules.length_regulator import (
            InterpolateRegulator,
        )

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=(args.length_regulator.in_channels if hasattr(args.length_regulator, "in_channels") else None),
            vector_quantize=(
                args.length_regulator.vector_quantize if hasattr(args.length_regulator, "vector_quantize") else False
            ),
            codebook_size=args.length_regulator.content_codebook_size,
            n_codebooks=(args.length_regulator.n_codebooks if hasattr(args.length_regulator, "n_codebooks") else 1),
            quantizer_dropout=(
                args.length_regulator.quantizer_dropout if hasattr(args.length_regulator, "quantizer_dropout") else 0.0
            ),
            f0_condition=(
                args.length_regulator.f0_condition if hasattr(args.length_regulator, "f0_condition") else False
            ),
            n_f0_bins=(args.length_regulator.n_f0_bins if hasattr(args.length_regulator, "n_f0_bins") else 512),
        )

        if use_gpt_latent:
            self.models = nn.ModuleDict(
                {
                    "cfm": CFM(args),
                    "length_regulator": length_regulator,
                    "gpt_layer": torch.nn.Sequential(
                        torch.nn.Linear(1280, 256),
                        torch.nn.Linear(256, 128),
                        torch.nn.Linear(128, 1024),
                    ),
                }
            )
        else:
            self.models = nn.ModuleDict({"cfm": CFM(args), "length_regulator": length_regulator})

    def forward(self, x, target_lengths, prompt_len, cond, y):
        x = self.models["cfm"](x, target_lengths, prompt_len, cond, y)
        return x

    def forward2(self, s_ori, target_lengths, f0_ori):
        x = self.models["length_regulator"](s_ori, ylens=target_lengths, f0=f0_ori)
        return x

    def forward_emovec(self, x):
        x = self.models["emo_layer"](x)
        return x

    def forward_emo_encoder(self, x):
        x = self.models["emo_encoder"](x)
        return x

    def forward_gpt(self, x):
        x = self.models["gpt_layer"](x)
        return x

    def enable_torch_compile(self):
        """Enable torch.compile optimization."""
        if "cfm" in self.models:
            self.models["cfm"].enable_torch_compile()


def load_checkpoint(
    model,
    optimizer,
    path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
    load_ema=False,
):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    if load_ema and "ema" in state:
        logger.info("Loading EMA")
        for key in model:
            i = 0
            for param_name in params[key]:
                if "input_pos" in param_name:
                    continue
                assert params[key][param_name].shape == state["ema"][key][0][i].shape
                params[key][param_name] = state["ema"][key][0][i].clone()
                i += 1
    for key in model:
        if key in params and key not in ignore_modules:
            if not is_distributed:
                for k in list(params[key].keys()):
                    if k.startswith("module."):
                        params[key][k[len("module.") :]] = params[key][k]
                        del params[key][k]
            model_state_dict = model[key].state_dict()
            filtered_state_dict = {
                k: v for k, v in params[key].items() if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                logger.warning("Skipped loading some keys due to shape mismatch: %s", skipped_keys)
            logger.debug("%s loaded", key)
            model[key].load_state_dict(filtered_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])
    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters


def load_checkpoint2(
    model,
    optimizer,
    path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
    load_ema=False,
):
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    if load_ema and "ema" in state:
        logger.info("Loading EMA")
        for key in model.models:
            i = 0
            for param_name in params[key]:
                if "input_pos" in param_name:
                    continue
                assert params[key][param_name].shape == state["ema"][key][0][i].shape
                params[key][param_name] = state["ema"][key][0][i].clone()
                i += 1
    for key in model.models:
        if key in params and key not in ignore_modules:
            if not is_distributed:
                for k in list(params[key].keys()):
                    if k.startswith("module."):
                        params[key][k[len("module.") :]] = params[key][k]
                        del params[key][k]
            model_state_dict = model.models[key].state_dict()
            filtered_state_dict = {
                k: v for k, v in params[key].items() if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                logger.warning("Skipped loading some keys due to shape mismatch: %s", skipped_keys)
            logger.debug("%s loaded", key)
            model.models[key].load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])
    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters
