# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from FunCineForge Demo (BSD-3-Clause)
# Original: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
"""Ultra-Light face detector using ONNX Runtime."""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch


def _hard_nms(
    box_scores: torch.Tensor,
    iou_threshold: float,
    top_k: int = -1,
    candidate_size: int = 200,
) -> torch.Tensor:
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked: list[int] = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        overlap_lt = torch.max(rest_boxes[..., :2], current_box[:2])
        overlap_rb = torch.min(rest_boxes[..., 2:], current_box[2:])
        overlap_hw = torch.clamp(overlap_rb - overlap_lt, min=0.0)
        inter = overlap_hw[..., 0] * overlap_hw[..., 1]
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        area_cur = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        iou = inter / (area_rest + area_cur - inter + 1e-5)
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


class FaceDetPredictor:
    """ONNX-based Ultra-Light face detector (version-RFB-320)."""

    _IMAGE_SIZE = (320, 240)
    _IMAGE_MEAN = np.array([127, 127, 127], dtype=np.float32)
    _IMAGE_STD = 128.0

    def __init__(self, model_dir: str, device: str = "cpu", device_id: int = 0):
        import onnxruntime

        onnx_path = os.path.join(model_dir, "version-RFB-320.onnx")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"Face detection model not found: {onnx_path}")
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": device_id}))
        self._session = onnxruntime.InferenceSession(onnx_path, opts, providers=providers)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, self._IMAGE_SIZE)
        img = img.astype(np.float32)
        img -= self._IMAGE_MEAN
        img /= self._IMAGE_STD
        return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

    def __call__(
        self,
        image: np.ndarray,
        top_k: int = -1,
        prob_threshold: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = image.shape[:2]
        tensor = self._preprocess(image)
        scores_raw, boxes_raw = self._session.run(
            None, {self._session.get_inputs()[0].name: tensor}
        )
        boxes = torch.from_numpy(boxes_raw[0])
        scores = torch.from_numpy(scores_raw[0])

        picked_box_probs: list[torch.Tensor] = []
        picked_labels: list[int] = []
        for cls in range(1, scores.size(1)):
            probs = scores[:, cls]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset = boxes[mask, :]
            bp = torch.cat([subset, probs.reshape(-1, 1)], dim=1)
            bp = _hard_nms(bp, 0.3, top_k, candidate_size=200)
            picked_box_probs.append(bp)
            picked_labels.extend([cls] * bp.size(0))

        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        result = torch.cat(picked_box_probs)
        result[:, 0] *= width
        result[:, 1] *= height
        result[:, 2] *= width
        result[:, 3] *= height
        return result[:, :4], torch.tensor(picked_labels), result[:, 4]
