"""
Frame pre-processing for varying lighting conditions.

* Automatic gamma boost for very dark frames (night driving, tunnels).
* CLAHE on the luminance channel to even out harsh side lighting / shadows.
Only the frame given to the landmark detector is modified; the dashboard shows it too,
so the user can see what the detector "sees".
"""
from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, cfg):
        self.use_clahe = bool(cfg.clahe)
        self.auto_gamma = bool(cfg.auto_gamma)
        self.dark_threshold = float(cfg.dark_threshold)
        self._clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=(8, 8))
        self._gamma_luts: dict[float, np.ndarray] = {}
        self.last_brightness = 0.0

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        key = round(gamma, 1)
        if key not in self._gamma_luts:
            inv = 1.0 / key
            self._gamma_luts[key] = np.array(
                [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
        return self._gamma_luts[key]

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_brightness = float(gray.mean())
        out = frame
        if self.auto_gamma and self.last_brightness < self.dark_threshold:
            # darker frame -> stronger boost (gamma 1.2 .. 2.5)
            gamma = 1.2 + 1.3 * (1.0 - self.last_brightness / max(self.dark_threshold, 1.0))
            out = cv2.LUT(out, self._gamma_lut(gamma))
        if self.use_clahe:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe.apply(l_ch)
            out = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return out
