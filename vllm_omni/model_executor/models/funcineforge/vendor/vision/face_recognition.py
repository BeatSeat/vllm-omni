# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from FunCineForge Demo (BSD-3-Clause)
# Original: https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
"""CurricularFace IR101 face recognition using ONNX Runtime.

Outputs 512-dim unnormalized embeddings that preserve expression intensity.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class FaceRecIR101:
    """ONNX-based face recognition embedding extractor (512-dim)."""

    def __init__(self, model_dir: str, device: str = "cpu", device_id: int = 0):
        import onnxruntime

        onnx_path = os.path.join(model_dir, "face_recog_ir101.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"Face recognition model not found: {onnx_path}")
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": device_id}))
        self._session = onnxruntime.InferenceSession(onnx_path, opts, providers=providers)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img[:, :, ::-1]  # BGR -> RGB
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = (img / 255.0 - 0.5) / 0.5
        img = np.expand_dims(img.astype(np.float32), 0)
        return self._session.run(None, {self._session.get_inputs()[0].name: img})[0]
