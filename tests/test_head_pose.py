"""Head-pose sign convention: pitching the head DOWN must give a positive pitch."""
import unittest

import cv2
import numpy as np

from src.head_pose import MODEL_POINTS, POSE_LANDMARKS, HeadPoseEstimator


def synthetic_landmarks(pitch_rad=0.0, yaw_rad=0.0):
    est = HeadPoseEstimator()
    cam = est.camera_matrix(640, 480)
    rx, _ = cv2.Rodrigues(np.array([pitch_rad, 0.0, 0.0]))
    ry, _ = cv2.Rodrigues(np.array([0.0, yaw_rad, 0.0]))
    pts = (ry @ rx @ MODEL_POINTS.T).T
    img, _ = cv2.projectPoints(pts, np.zeros(3), np.array([0, 0, 2000.0]), cam, np.zeros(4))
    lm = np.zeros((478, 2))
    lm[POSE_LANDMARKS] = img.reshape(-1, 2)
    return lm


class TestHeadPose(unittest.TestCase):
    def test_frontal(self):
        pitch, yaw, _ = HeadPoseEstimator().estimate(synthetic_landmarks(), (480, 640))
        self.assertAlmostEqual(pitch, 0, delta=1)
        self.assertAlmostEqual(yaw, 0, delta=1)

    def test_head_down_positive(self):
        pitch, _, _ = HeadPoseEstimator().estimate(synthetic_landmarks(pitch_rad=0.4), (480, 640))
        self.assertAlmostEqual(pitch, np.degrees(0.4), delta=1.5)

    def test_invert_option(self):
        pitch, _, _ = HeadPoseEstimator(invert_pitch=True).estimate(
            synthetic_landmarks(pitch_rad=0.4), (480, 640))
        self.assertLess(pitch, 0)


if __name__ == "__main__":
    unittest.main()
