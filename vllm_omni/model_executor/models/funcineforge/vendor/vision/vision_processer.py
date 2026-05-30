# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from FunCineForge Demo (BSD-3-Clause)
"""Video face embedding extraction pipeline.

Processes video frames to extract per-frame 512-dim face embeddings with
active speaker detection filtering. Pipeline stages:
  1. Face detection (Ultra-Light ONNX)
  2. IoU-based face tracking
  3. Active speaker detection (TalkNet ONNX)
  4. Lip landmark detection (FAN PyTorch)
  5. Face recognition embedding (IR101 ONNX)
"""

from __future__ import annotations

import gc
import pickle
import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.funcineforge.vendor.vision import VisualFrontend

_OFF_THRESHOLD = -0.5


class VisionProcesser:
    """Extract face embeddings from a video clip aligned to audio VAD segments."""

    def __init__(
        self,
        video_file_path: str,
        audio_file_path: str,
        audio_vad: list[list[float]],
        out_feat_path: str,
        visual_models: VisualFrontend,
        conf: dict | None = None,
    ):
        fs, audio = wavfile.read(audio_file_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        duration = audio.shape[0] / fs
        target_length = int(duration * 16000)
        self.audio = signal.resample(audio, target_length)

        self.audio_vad = [[int(i * 16000), int(j * 16000)] for (i, j) in audio_vad]
        self.video_path = video_file_path
        self.cap = cv2.VideoCapture(video_file_path)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.visual_models = visual_models
        self.out_feat_path = out_feat_path

        conf = conf or {}
        self.min_track = conf.get("min_track", 10)
        self.num_failed_det = conf.get("num_failed_det", 10)
        self.crop_scale = conf.get("crop_scale", 0.4)
        self.min_face_size = conf.get("min_face_size", 1)
        self.face_det_stride = conf.get("face_det_stride", 5)
        self.shot_stride = conf.get("shot_stride", 50)

        self._result: dict = {
            "frameI": np.empty((0,), dtype=np.int32),
            "feat": np.empty((0, 512), dtype=np.float32),
            "faceI": np.empty((0,), dtype=np.int32),
            "face": [],
            "face_bbox": np.empty((0, 4), dtype=np.int32),
            "lip": [],
            "lip_bbox": np.empty((0, 4), dtype=np.int32),
        }

    def run(self) -> None:
        frames: list[np.ndarray] = []
        face_det_frames: list[np.ndarray] = []

        for audio_st, audio_ed in self.audio_vad:
            frame_st = int(audio_st / 640)
            frame_ed = int(audio_ed / 640)
            num_frames = frame_ed - frame_st + 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_st)

            for idx in range(num_frames):
                ret, frame = self.cap.read()
                if not ret:
                    break
                if idx % self.face_det_stride == 0:
                    face_det_frames.append(frame)
                frames.append(frame)
                if (idx + 1) % self.shot_stride == 0:
                    audio_slice = self.audio[
                        (frame_st + idx + 1 - self.shot_stride) * 640 : (frame_st + idx + 1) * 640
                    ]
                    self._process_shot(frames, face_det_frames, audio_slice, frame_st + idx + 1 - self.shot_stride)
                    frames, face_det_frames = [], []

            if frames:
                offset = frame_st + num_frames - len(frames)
                audio_slice = self.audio[offset * 640 : (frame_st + num_frames) * 640]
                self._process_shot(frames, face_det_frames, audio_slice, offset)
                frames, face_det_frames = [], []

        self.cap.release()
        out_data = {
            "embeddings": self._result["feat"],
            "frameI": self._result["frameI"],
            "faceI": self._result["faceI"],
            "face": self._result["face"],
            "face_bbox": self._result["face_bbox"],
            "lip": self._result["lip"],
            "lip_bbox": self._result["lip_bbox"],
        }
        with open(self.out_feat_path, "wb") as f:
            pickle.dump(out_data, f)

    def close(self) -> None:
        try:
            if hasattr(self, "_result"):
                for v in self._result.values():
                    if isinstance(v, list):
                        v.clear()
                self._result.clear()
        except Exception:
            pass
        gc.collect()

    # -- internal pipeline stages --

    def _process_shot(
        self,
        frames: list[np.ndarray],
        face_det_frames: list[np.ndarray],
        audio: np.ndarray,
        frame_st: int,
    ) -> None:
        dets = self._detect_faces(face_det_frames)
        tracks = self._track_shot(dets)
        vid_tracks = [self._crop_video(t, frames, audio) for t in tracks]
        scores = self._evaluate_asd(vid_tracks)
        embs = self._evaluate_fr(frames, vid_tracks, scores)

        self._result["frameI"] = np.append(self._result["frameI"], embs["frameI"] + frame_st)
        self._result["feat"] = np.append(self._result["feat"], embs["feat"], axis=0)
        self._result["faceI"] = np.append(self._result["faceI"], embs["faceI"] + frame_st)
        self._result["face"].extend(embs["face"])
        self._result["face_bbox"] = np.vstack([self._result["face_bbox"], embs["face_bbox"]])
        self._result["lip"].extend(embs["lip"])
        self._result["lip_bbox"] = np.vstack([self._result["lip_bbox"], embs["lip_bbox"]])

    def _detect_faces(self, frames: list[np.ndarray]) -> list[list[dict]]:
        import torch

        dets: list[list[dict]] = []
        for fidx, image in enumerate(frames):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, _, probs = self.visual_models.detect_faces(rgb, top_k=10, prob_threshold=0.9)
            bboxes = torch.cat([bboxes, probs.reshape(-1, 1)], dim=-1)
            frame_dets: list[dict] = []
            for bbox in bboxes:
                frame_dets.append({
                    "frame": fidx * self.face_det_stride,
                    "bbox": bbox[:-1].tolist(),
                    "conf": bbox[-1].item(),
                })
            dets.append(frame_dets)
        return dets

    def _track_shot(self, scene_faces: list[list[dict]]) -> list[dict]:
        tracks: list[dict] = []
        while True:
            track: list[dict] = []
            for frame_faces in scene_faces:
                for face in frame_faces:
                    if not track:
                        track.append(face)
                        frame_faces.remove(face)
                        break
                    elif face["frame"] - track[-1]["frame"] <= self.num_failed_det:
                        iou = self._iou(face["bbox"], track[-1]["bbox"])
                        if iou > 0.5:
                            track.append(face)
                            frame_faces.remove(face)
                            break
                    else:
                        break
            if not track:
                break
            if len(track) > 1 and track[-1]["frame"] - track[0]["frame"] + 1 >= self.min_track:
                frame_num = np.array([f["frame"] for f in track])
                bboxes = np.array([f["bbox"] for f in track])
                frame_i = np.arange(frame_num[0], frame_num[-1] + 1)
                bboxes_i = np.stack(
                    [interp1d(frame_num, bboxes[:, j])(frame_i) for j in range(4)], axis=1
                )
                if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > self.min_face_size:
                    tracks.append({"frame": frame_i, "bbox": bboxes_i})
        return tracks

    @staticmethod
    def _iou(box_a: list[float], box_b: list[float]) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0

    def _crop_video(self, track: dict, frames: list[np.ndarray], audio: np.ndarray) -> dict:
        crop_frames: list[np.ndarray] = []
        dets: dict[str, list[float]] = {"x": [], "y": [], "s": []}
        for det in track["bbox"]:
            dets["s"].append(max(det[3] - det[1], det[2] - det[0]) / 2)
            dets["y"].append((det[1] + det[3]) / 2)
            dets["x"].append((det[0] + det[2]) / 2)

        cs = self.crop_scale
        for fidx, frame_idx in enumerate(track["frame"]):
            bs = dets["s"][fidx]
            bsi = int(bs * (1 + 2 * cs))
            image = frames[frame_idx]
            padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110)
            my = dets["y"][fidx] + bsi
            mx = dets["x"][fidx] + bsi
            face = padded[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]
            crop_frames.append(cv2.resize(face, (224, 224)))

        crop_audio = audio[int(track["frame"][0]) * 640 : int(track["frame"][-1] + 1) * 640]
        return {"track": track, "proc_track": dets, "data": [crop_frames, crop_audio]}

    def _evaluate_asd(self, tracks: list[dict]) -> list[np.ndarray]:
        import python_speech_features

        all_scores: list[np.ndarray] = []
        for ins in tracks:
            video, audio = ins["data"]
            audio_feat = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
            video_feat = []
            for frame in video:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h0, w0 = gray.shape
                interp = cv2.INTER_CUBIC if (h0 < 224 or w0 < 224) else cv2.INTER_AREA
                gray = cv2.resize(gray, (224, 224), interpolation=interp)
                gray = gray[56:168, 56:168]
                video_feat.append(gray)
            video_feat_arr = np.array(video_feat)

            length = min(
                (audio_feat.shape[0] - audio_feat.shape[0] % 4) / 100,
                video_feat_arr.shape[0] / 25,
            )
            audio_feat = audio_feat[: int(round(length * 100)), :]
            video_feat_arr = video_feat_arr[: int(round(length * 25)), :, :]

            score = self.visual_models.asd_score(
                np.expand_dims(audio_feat, 0).astype(np.float32),
                np.expand_dims(video_feat_arr, 0).astype(np.float32),
            )
            all_scores.append(np.asarray(score, dtype=np.float32))
        return all_scores

    def _evaluate_fr(
        self,
        frames: list[np.ndarray],
        tracks: list[dict],
        scores: list[np.ndarray],
    ) -> dict:
        smooth_scores: list[np.ndarray] = []
        for score in scores:
            s = np.asarray(score).flatten()
            if s.size == 0:
                smooth_scores.append(s)
                continue
            s_med = signal.medfilt(s, kernel_size=min(5, len(s) | 1))
            s_avg = np.convolve(s_med, np.ones(5) / 5, mode="same")
            smooth_scores.append(s_avg)

        faces_per_frame: list[list[dict]] = [[] for _ in range(len(frames))]
        for tidx, track_data in enumerate(tracks):
            sc = smooth_scores[tidx]
            for fidx, frame_idx in enumerate(track_data["track"]["frame"].tolist()):
                s_window = sc[max(fidx - 4, 0) : min(fidx + 5, len(sc))]
                s = float(np.mean(s_window))
                bbox = track_data["track"]["bbox"][fidx].astype(np.int32)
                face = frames[frame_idx][
                    max(bbox[1], 0) : min(bbox[3], frames[frame_idx].shape[0]),
                    max(bbox[0], 0) : min(bbox[2], frames[frame_idx].shape[1]),
                ]
                faces_per_frame[frame_idx].append({"track": tidx, "score": s, "facedata": face, "bbox": bbox})

        result: dict[str, list] = {
            "frameI": [], "faceI": [], "feat": [],
            "face": [], "face_bbox": [], "lip": [], "lip_bbox": [],
        }
        for fidx in range(0, len(faces_per_frame), max(1, self.face_det_stride)):
            if not faces_per_frame[fidx]:
                continue
            best = max(faces_per_frame[fidx], key=lambda x: x["score"])
            lip_res = self.visual_models.detect_lip(best["facedata"])
            if lip_res is None or lip_res.get("lip_crop") is None:
                continue
            result["faceI"].append(fidx)
            result["face"].append(best["facedata"])
            result["lip"].append(lip_res["lip_crop"])
            result["face_bbox"].append(best["bbox"])
            result["lip_bbox"].append(lip_res["lip_bbox"])
            feature = self.visual_models.get_face_embedding(best["facedata"])
            result["feat"].append(feature)
            if best["score"] >= _OFF_THRESHOLD:
                result["frameI"].append(fidx)

        return {
            "frameI": np.array(result["frameI"], dtype=np.int32),
            "faceI": np.array(result["faceI"], dtype=np.int32),
            "feat": np.vstack(result["feat"]) if result["feat"] else np.empty((0, 512), np.float32),
            "face": result["face"],
            "face_bbox": np.array(result["face_bbox"], dtype=np.int32) if result["face_bbox"] else np.empty((0, 4), np.int32),
            "lip": result["lip"],
            "lip_bbox": np.array(result["lip_bbox"], dtype=np.int32) if result["lip_bbox"] else np.empty((0, 4), np.int32),
        }
