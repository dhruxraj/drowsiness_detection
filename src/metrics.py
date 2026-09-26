"""
Geometric drowsiness metrics computed from MediaPipe Face Mesh landmarks.

* EAR - Eye Aspect Ratio   (Soukupova & Cech, 2016)
* MAR - Mouth Aspect Ratio (same idea applied to the inner lip contour)

All functions take landmark coordinates in PIXELS (not normalised 0..1),
because normalised x and y are scaled differently when the frame is not square.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (468/478-point topology)
# Order for EAR: p1 (corner), p2, p3 (upper lid), p4 (other corner), p5, p6 (lower lid)
#
#          p2    p3
#     p1 ------------ p4
#          p6    p5
# ---------------------------------------------------------------------------
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]   # driver's right eye (image left when not mirrored)
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]   # driver's left eye

# Order for MAR (inner lips): p1 left corner, p2-p4 upper lip, p5 right corner, p6-p8 lower lip
#          p2  p3  p4
#     p1 --------------- p5
#          p8  p7  p6
MOUTH_MAR = [61, 81, 13, 311, 291, 402, 14, 178]

# Contours used only for drawing on the dashboard
RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
LEFT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
INNER_LIPS_CONTOUR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                      324, 318, 402, 317, 14, 87, 178, 88, 95]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    ~0.25-0.35 for an open eye, drops towards ~0.05-0.15 when the eye closes.
    It is invariant to face size/distance because it is a ratio.
    """
    p1, p2, p3, p4, p5, p6 = eye
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """
    MAR = (|p2 - p8| + |p3 - p7| + |p4 - p6|) / (2 * |p1 - p5|)

    ~0.0-0.1 closed mouth, ~0.2-0.5 talking, > ~0.6 wide open (yawn).
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth
    horizontal = _dist(p1, p5)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p8) + _dist(p3, p7) + _dist(p4, p6)) / (2.0 * horizontal)


def compute_ear(landmarks: np.ndarray) -> float:
    """Average EAR of both eyes (averaging reduces noise and the effect of a wink)."""
    right = eye_aspect_ratio(landmarks[RIGHT_EYE_EAR])
    left = eye_aspect_ratio(landmarks[LEFT_EYE_EAR])
    return (right + left) / 2.0


def compute_mar(landmarks: np.ndarray) -> float:
    return mouth_aspect_ratio(landmarks[MOUTH_MAR])


class EMA:
    """Exponential moving average used to smooth noisy per-frame measurements."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None
