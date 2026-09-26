"""
Facial landmark detection with MediaPipe (478 landmarks incl. iris).

MediaPipe offers two APIs and recent releases (>= ~0.10.2x) removed the old one:
  * "legacy"  -> mp.solutions.face_mesh     (no model file needed)
  * "tasks"   -> mediapipe.tasks FaceLandmarker (needs models/face_landmarker.task,
                 download with:  python tools/download_model.py)
engine: auto picks legacy if available, otherwise tasks.
Both return the same landmark topology, so the rest of the system is unchanged.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


class LandmarkDetectorError(RuntimeError):
    pass


class FaceLandmarkDetector:
    def __init__(self, cfg):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise LandmarkDetectorError("mediapipe is not installed: pip install -r requirements.txt") from exc
        self._mp = mp
        engine = str(cfg.engine).lower()
        has_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
        if engine == "auto":
            engine = "legacy" if has_legacy else "tasks"
        self.engine = engine

        if engine == "legacy":
            if not has_legacy:
                raise LandmarkDetectorError(
                    "This MediaPipe version has no mp.solutions. Set face_mesh.engine: tasks "
                    "and run  python tools/download_model.py")
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=int(cfg.max_num_faces),
                refine_landmarks=bool(cfg.refine_landmarks),
                min_detection_confidence=float(cfg.min_detection_confidence),
                min_tracking_confidence=float(cfg.min_tracking_confidence),
            )
        elif engine == "tasks":
            path = cfg.task_model_path
            if not os.path.isfile(path):
                raise LandmarkDetectorError(
                    f"Face landmarker model not found at '{path}'. Run:  python tools/download_model.py")
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=int(cfg.max_num_faces),
                min_face_detection_confidence=float(cfg.min_detection_confidence),
                min_face_presence_confidence=float(cfg.min_detection_confidence),
                min_tracking_confidence=float(cfg.min_tracking_confidence),
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._last_ts_ms = -1
        else:
            raise LandmarkDetectorError(f"Unknown face_mesh.engine '{engine}' (use auto, legacy or tasks)")

    @staticmethod
    def _largest(faces, w: int, h: int) -> np.ndarray:
        """Convert faces to pixel arrays and return the largest one (= the driver)."""
        best, best_area = None, -1.0
        for lms in faces:
            pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
            span = pts.max(axis=0) - pts.min(axis=0)
            area = float(span[0] * span[1])
            if area > best_area:
                best, best_area = pts, area
        return best

    def detect(self, frame_bgr: np.ndarray, timestamp_s: float) -> np.ndarray | None:
        """Return an (N, 2) array of landmark pixel coordinates, or None if no face."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.engine == "legacy":
            rgb.flags.writeable = False
            res = self._mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            return self._largest([f.landmark for f in res.multi_face_landmarks], w, h)

        # Tasks API needs strictly increasing integer millisecond timestamps
        ts_ms = int(timestamp_s * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect_for_video(image, ts_ms)
        if not res.face_landmarks:
            return None
        return self._largest(res.face_landmarks, w, h)

    def close(self) -> None:
        if self.engine == "legacy":
            self._mesh.close()
        else:
            self._landmarker.close()
