# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from FunCineForge Demo (BSD-3-Clause)
# Modified face-alignment API (v1.4.1): TorchScript loading + local weight paths.
# Original face-alignment: https://github.com/1adrianb/face-alignment (BSD-3-Clause)
"""FAN-based lip landmark detector with bundled FaceAlignment loader."""

from __future__ import annotations

import os
import warnings
from enum import IntEnum

import cv2
import numpy as np
import torch


class LandmarksType(IntEnum):
    TWO_D = 1
    TWO_HALF_D = 2
    THREE_D = 3


class _FaceAlignmentLocal:
    """Minimal FaceAlignment using TorchScript models from local paths.

    This replaces the full ``face_alignment.api.FaceAlignment`` class.
    Only 2D landmark detection is supported (sufficient for lip extraction).
    """

    def __init__(
        self,
        device: str = "cpu",
        net_path: str | None = None,
        path_to_detector: str | None = None,
    ):
        from face_alignment.detection import sfd
        from face_alignment.utils import crop, get_preds_fromhm

        self._crop = crop
        self._get_preds_fromhm = get_preds_fromhm
        self.device = device

        self.face_detector = sfd.FaceDetector(
            device=device,
            verbose=False,
            **({"path_to_detector": path_to_detector} if path_to_detector else {}),
        )

        if net_path is None:
            raise ValueError("net_path (path to FAN TorchScript .zip) is required")
        self.face_alignment_net = torch.jit.load(net_path)
        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

    @torch.no_grad()
    def get_landmarks(self, image: np.ndarray) -> list[np.ndarray] | None:
        detected_faces = self.face_detector.detect_from_image(image.copy())
        if len(detected_faces) == 0:
            return None

        results: list[np.ndarray] = []
        for det in detected_faces:
            d = det[:4]
            center = torch.tensor(
                [(d[2] + d[0]) / 2.0, (d[3] + d[1]) / 2.0],
                dtype=torch.float32,
            )
            center[1] -= (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / 195.0

            inp = self._crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(2, 0, 1)).float()
            inp = inp.to(self.device).unsqueeze(0) / 255.0

            out = self.face_alignment_net(inp)[-1]
            if out.dim() == 4:
                out = out[:, -68:]

            pts, pts_img = self._get_preds_fromhm(out, center.unsqueeze(0), scale)
            pts_img = pts_img.view(68, 2).cpu().numpy()
            results.append(pts_img)

        return results if results else None


class LipDetector:
    """Detects lip region from a face crop using FAN landmarks.

    Requires ``face_alignment`` pip package for its SFD detector and
    utility functions. The FAN network itself is loaded from TorchScript
    weights bundled with the FunCineForge model.
    """

    def __init__(self, model_dir: str, device: str = "cpu", device_id: int = 0):
        device_str = f"cuda:{device_id}" if device == "cuda" else "cpu"
        fan_pth = os.path.join(model_dir, "fun_2d.pth")
        fan_zip = os.path.join(model_dir, "fun_2d.zip")

        if not os.path.isfile(fan_zip):
            raise FileNotFoundError(
                f"FAN TorchScript model not found: {fan_zip}. "
                "Ensure FunAudioLLM/Fun-CineForge model is fully downloaded."
            )

        try:
            self._fa = _FaceAlignmentLocal(
                device=device_str,
                net_path=fan_zip,
                path_to_detector=fan_pth if os.path.isfile(fan_pth) else None,
            )
        except ImportError as exc:
            raise ImportError(
                "FunCineForge lip detection requires the face-alignment package. "
                "Install with: pip install face-alignment>=1.4"
            ) from exc

    def detect_lip(self, face_img: np.ndarray) -> dict | None:
        h, w = face_img.shape[:2]
        try:
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            preds = self._fa.get_landmarks(rgb)
            if preds is None or len(preds) == 0:
                return None

            kps = preds[0]
            mouth_kps = kps[48:68]
            min_xy = mouth_kps.min(axis=0)
            max_xy = mouth_kps.max(axis=0)
            pad = 0.18 * (max_xy - min_xy)
            x1 = int(max(0, min_xy[0] - pad[0]))
            y1 = int(max(0, min_xy[1] - pad[1]))
            x2 = int(min(w - 1, max_xy[0] + pad[0]))
            y2 = int(min(h - 1, max_xy[1] + pad[1]))
            lip_bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
            lip_crop = face_img[y1:y2, x1:x2].copy()
            return {"lip_bbox": lip_bbox, "lip_crop": lip_crop, "kps": mouth_kps}
        except Exception:
            warnings.warn("FAN lip detection failed, skipping frame", stacklevel=2)
            return None
