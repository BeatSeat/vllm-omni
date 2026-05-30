# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from FunCineForge Demo (BSD-3-Clause)
# Original: https://github.com/TaoRuijie/TalkNet-ASD
"""TalkNet active speaker detection using ONNX Runtime."""

from __future__ import annotations

import os

import numpy as np


class ASDTalknet:
    """ONNX-based TalkNet active speaker detector."""

    def __init__(self, model_dir: str, device: str = "cpu", device_id: int = 0):
        import onnxruntime

        onnx_path = os.path.join(model_dir, "asd.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ASD model not found: {onnx_path}")
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": device_id}))
        self._session = onnxruntime.InferenceSession(onnx_path, opts, providers=providers)

    def __call__(self, audio_feat: np.ndarray, video_feat: np.ndarray) -> np.ndarray:
        inputs = {
            self._session.get_inputs()[0].name: audio_feat,
            self._session.get_inputs()[1].name: video_feat,
        }
        return self._session.run(None, inputs)[0]
