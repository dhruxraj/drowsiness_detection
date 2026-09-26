"""
Per-driver calibration.

For `calibration.duration` seconds the driver looks straight ahead, eyes open, mouth closed.
From these frames we compute:
  * open-eye EAR     -> personal closed-eye threshold = ear_ratio x open EAR
                        (eye shape differs between people; a fixed 0.21 is not ideal for all)
  * resting MAR      -> yawn threshold = max(configured threshold, resting MAR + margin)
  * neutral pitch/yaw -> head angles are measured relative to this pose, which removes the
                        effect of camera height/angle and of the generic 3-D face model.
If too few valid frames are collected (no face), the configured defaults are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .temporal_analyzer import FrameMetrics


@dataclass
class CalibrationResult:
    ok: bool
    ear_threshold: float
    mar_threshold: float
    pitch0: float
    yaw0: float
    open_ear: float | None = None
    rest_mar: float | None = None
    samples: int = 0

    @property
    def message(self) -> str:
        if not self.ok:
            return (f"CALIBRATION FAILED ({self.samples} samples) - using defaults "
                    f"EAR<{self.ear_threshold:.3f}, MAR>{self.mar_threshold:.2f}")
        return (f"CALIBRATION OK ({self.samples} samples): open EAR {self.open_ear:.3f} -> "
                f"threshold {self.ear_threshold:.3f}; rest MAR {self.rest_mar:.2f} -> "
                f"yawn threshold {self.mar_threshold:.2f}; neutral pitch {self.pitch0:+.1f}, "
                f"yaw {self.yaw0:+.1f}")


class Calibrator:
    def __init__(self, cfg):
        self.c, self.e, self.m = cfg.calibration, cfg.eyes, cfg.mouth
        self.start_time: float | None = None
        self._samples: list[FrameMetrics] = []
        self.done = False

    def start(self) -> None:
        self.start_time = None
        self._samples = []
        self.done = False

    def progress(self, t: float) -> float:
        if self.start_time is None:
            return 0.0
        return min(1.0, (t - self.start_time) / self.c.duration)

    def add(self, m: FrameMetrics | None, t: float) -> None:
        if self.done:
            return
        if self.start_time is None:
            self.start_time = t
        if m is not None:
            self._samples.append(m)
        if t - self.start_time >= self.c.duration:
            self.done = True

    def result(self) -> CalibrationResult:
        n = len(self._samples)
        if n < self.c.min_samples:
            return CalibrationResult(False, float(self.e.ear_threshold), float(self.m.mar_threshold),
                                     0.0, 0.0, samples=n)
        ears = np.array([s.ear for s in self._samples])
        # the upper 70 % of EAR values excludes blinks that happened during calibration
        open_ear = float(np.median(ears[ears >= np.percentile(ears, 30)]))
        ear_thr = float(np.clip(self.c.ear_ratio * open_ear, self.c.ear_min, self.c.ear_max))
        rest_mar = float(np.median([s.mar for s in self._samples]))
        mar_thr = max(float(self.m.mar_threshold), rest_mar + float(self.c.mar_margin))
        pitches = [s.pitch for s in self._samples if s.pitch is not None]
        yaws = [s.yaw for s in self._samples if s.yaw is not None]
        pitch0 = float(np.median(pitches)) if pitches else 0.0
        yaw0 = float(np.median(yaws)) if yaws else 0.0
        return CalibrationResult(True, ear_thr, mar_thr, pitch0, yaw0, open_ear, rest_mar, n)
