# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored visual preprocessing models for FunCineForge video dubbing.

All ONNX/PyTorch weights are expected in the model directory
(downloaded via ``snapshot_download("FunAudioLLM/Fun-CineForge")``).
"""

from __future__ import annotations

from vllm_omni.model_executor.models.funcineforge.vendor.vision.asd import ASDTalknet
from vllm_omni.model_executor.models.funcineforge.vendor.vision.face_detection import FaceDetPredictor
from vllm_omni.model_executor.models.funcineforge.vendor.vision.face_recognition import FaceRecIR101
from vllm_omni.model_executor.models.funcineforge.vendor.vision.lip_detection import LipDetector

_VISION_CONF_DEFAULTS = {
    "min_track": 10,
    "num_failed_det": 10,
    "crop_scale": 0.4,
    "min_face_size": 1,
    "face_det_stride": 5,
    "shot_stride": 50,
}


class VisualFrontend:
    """Lightweight visual model manager that loads all weights from model_dir.

    Replaces the demo's ``GlobalModels`` singleton + ``ModelPool`` pattern
    with direct model instances. Thread safety is not needed because video
    preprocessing runs in a single-threaded executor per request.
    """

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.face_det = FaceDetPredictor(model_dir, device=device)
        self.asd = ASDTalknet(model_dir, device=device)
        self.face_rec = FaceRecIR101(model_dir, device=device)
        self.lip_det = LipDetector(model_dir, device=device)
        self.conf = dict(_VISION_CONF_DEFAULTS)

    def detect_faces(self, image, **kwargs):
        return self.face_det(image, **kwargs)

    def asd_score(self, audio_feat, video_feat):
        return self.asd(audio_feat, video_feat)

    def get_face_embedding(self, face_image):
        return self.face_rec(face_image)

    def detect_lip(self, face_image):
        return self.lip_det.detect_lip(face_image)


__all__ = [
    "ASDTalknet",
    "FaceDetPredictor",
    "FaceRecIR101",
    "LipDetector",
    "VisualFrontend",
]
