"""
Head pose (pitch / yaw / roll) estimation with OpenCV solvePnP.

Six 2-D face landmarks are matched to a generic 3-D face model; solvePnP finds
the rotation that best projects the model onto the image.

The 3-D model uses the OpenCV camera convention (x right, y DOWN, z AWAY from the
camera), so a driver looking straight at the camera gives angles close to 0.
Sign convention of the returned angles:
    pitch > 0  : head tilted DOWN  (nodding / head dropping)
    yaw   != 0 : head turned left/right
Because the model is generic, absolute angles differ between people and camera
positions; the system therefore uses angles RELATIVE to a calibrated neutral pose.
"""
from __future__ import annotations

import cv2
import numpy as np

# landmark index -> 3-D model point (arbitrary units, nose tip at origin)
_MODEL = {
    1: (0.0, 0.0, 0.0),            # nose tip
    152: (0.0, 330.0, 65.0),       # chin
    33: (-225.0, -170.0, 135.0),   # eye outer corner on image-left (driver's right eye)
    263: (225.0, -170.0, 135.0),   # eye outer corner on image-right (driver's left eye)
    61: (-150.0, 150.0, 125.0),    # mouth corner, image-left
    291: (150.0, 150.0, 125.0),    # mouth corner, image-right
}
POSE_LANDMARKS = list(_MODEL.keys())
MODEL_POINTS = np.array(list(_MODEL.values()), dtype=np.float64)


class HeadPoseEstimator:
    def __init__(self, invert_pitch: bool = False):
        self.invert_pitch = invert_pitch
        self._rvec = None
        self._tvec = None

    @staticmethod
    def camera_matrix(width: int, height: int) -> np.ndarray:
        # Approximation: focal length ~ image width, principal point at the centre.
        f = float(width)
        return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)

    def estimate(self, landmarks: np.ndarray, frame_shape) -> tuple[float, float, float] | None:
        """Return (pitch, yaw, roll) in degrees, or None if the estimate failed."""
        h, w = frame_shape[:2]
        image_points = landmarks[POSE_LANDMARKS].astype(np.float64)
        cam = self.camera_matrix(w, h)
        dist = np.zeros((4, 1))
        use_guess = self._rvec is not None
        try:
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS, image_points, cam, dist,
                rvec=self._rvec.copy() if use_guess else None,
                tvec=self._tvec.copy() if use_guess else None,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok = False
        if not ok or tvec[2, 0] <= 0:      # face must be in front of the camera
            self._rvec = self._tvec = None
            return None
        self._rvec, self._tvec = rvec, tvec
        rot, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rot)
        pitch, yaw, roll = (float(a) for a in angles)
        if self.invert_pitch:
            pitch = -pitch
        return pitch, yaw, roll

    def reset(self) -> None:
        self._rvec = self._tvec = None
