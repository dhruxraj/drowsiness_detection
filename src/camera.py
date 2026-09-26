"""
Camera input.

* Live cameras (USB webcam, laptop camera) are read in a background thread so the
  processing loop always gets the NEWEST frame (no growing lag if processing is slow).
* Raspberry Pi camera module via picamera2 (backend: picamera2).
* Video files are read synchronously, frame by frame, and time-stamped with the video's
  own clock, so recorded test videos give the same timing results on any computer.
"""
from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, cfg):
        src = cfg.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.source = src
        self.backend = str(cfg.backend).lower()
        self.width, self.height, self.fps = int(cfg.width), int(cfg.height), int(cfg.fps)
        self.reconnect_attempts = int(cfg.reconnect_attempts)
        self.is_file = isinstance(src, str) and os.path.isfile(src)

        self._cap = None
        self._picam = None
        self._thread = None
        self._lock = threading.Lock()
        self._running = False
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._last_read_seq = -1
        self._file_frame_idx = 0
        self._file_fps = 30.0
        self.finished = False   # True when a video file ends or the camera is lost for good

    # ------------------------------------------------------------------ open
    def _open_opencv(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise CameraError(f"Cannot open camera/video source: {self.source!r}")
        if not self.is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self._file_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _open_picamera2(self) -> None:
        try:
            from picamera2 import Picamera2  # only available on Raspberry Pi OS
        except ImportError as exc:
            raise CameraError("picamera2 is not installed (sudo apt install python3-picamera2)") from exc
        self._picam = Picamera2()
        conf = self._picam.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"})  # RGB888 = BGR byte order
        self._picam.configure(conf)
        self._picam.start()
        time.sleep(1.0)  # let auto-exposure settle

    def start(self) -> "Camera":
        if self.backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_opencv()
        if not self.is_file:
            self._running = True
            self._thread = threading.Thread(target=self._reader, name="camera", daemon=True)
            self._thread.start()
        return self

    # ------------------------------------------------------------------ live capture thread
    def _grab(self):
        if self._picam is not None:
            return True, self._picam.capture_array()
        return self._cap.read()

    def _reader(self) -> None:
        failures = 0
        while self._running:
            ok, frame = self._grab()
            if not ok or frame is None:
                failures += 1
                if failures > 30:           # ~1 s of failed reads -> try to reconnect
                    if not self._reconnect():
                        self.finished = True
                        self._running = False
                        return
                    failures = 0
                time.sleep(0.03)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self._ts = time.monotonic()
                self._seq += 1

    def _reconnect(self) -> bool:
        for attempt in range(1, self.reconnect_attempts + 1):
            print(f"[camera] connection lost - reconnecting ({attempt}/{self.reconnect_attempts})")
            try:
                if self._picam is not None:      # Pi camera: restart the stream
                    self._picam.stop()
                    self._picam.start()
                    return True
                if self._cap is not None:
                    self._cap.release()
                self._open_opencv()
                return True
            except CameraError:
                time.sleep(1.0)
        print("[camera] could not reconnect")
        return False

    # ------------------------------------------------------------------ public API
    def read(self, timeout: float = 1.0) -> tuple[np.ndarray | None, float]:
        """Return (frame, timestamp_seconds). frame is None if nothing new arrived in time."""
        if self.is_file:
            ok, frame = self._cap.read()
            if not ok:
                self.finished = True
                return None, 0.0
            ts = self._file_frame_idx / self._file_fps
            self._file_frame_idx += 1
            return frame, ts

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.finished:
            with self._lock:
                if self._frame is not None and self._seq != self._last_read_seq:
                    self._last_read_seq = self._seq
                    return self._frame.copy(), self._ts
            time.sleep(0.002)
        return None, 0.0

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()
