"""Unit tests for EAR / MAR geometry."""
import unittest

import numpy as np

from src.metrics import EMA, eye_aspect_ratio, mouth_aspect_ratio


def eye(opening: float) -> np.ndarray:
    # corners 30 px apart, lids +-opening/2 around the centre line
    return np.array([[0, 0], [10, -opening / 2], [20, -opening / 2], [30, 0],
                     [20, opening / 2], [10, opening / 2]], dtype=float)


class TestEAR(unittest.TestCase):
    def test_open_eye(self):
        # vertical 9 px, horizontal 30 px -> (9 + 9) / (2 * 30) = 0.30
        self.assertAlmostEqual(eye_aspect_ratio(eye(9)), 0.30, places=6)

    def test_closed_eye(self):
        self.assertLess(eye_aspect_ratio(eye(1.5)), 0.1)

    def test_scale_invariant(self):
        self.assertAlmostEqual(eye_aspect_ratio(eye(9) * 3), eye_aspect_ratio(eye(9)), places=6)

    def test_degenerate(self):
        self.assertEqual(eye_aspect_ratio(np.zeros((6, 2))), 0.0)


class TestMAR(unittest.TestCase):
    def mouth(self, opening):
        w = 50.0
        return np.array([[0, 0], [12, -opening / 2], [25, -opening / 2], [38, -opening / 2],
                         [w, 0], [38, opening / 2], [25, opening / 2], [12, opening / 2]])

    def test_closed_mouth(self):
        self.assertAlmostEqual(mouth_aspect_ratio(self.mouth(0)), 0.0)

    def test_yawn(self):
        # 3 x 40 / (2 x 50) = 1.2
        self.assertAlmostEqual(mouth_aspect_ratio(self.mouth(40)), 1.2)


class TestEMA(unittest.TestCase):
    def test_smoothing(self):
        e = EMA(0.5)
        self.assertEqual(e.update(1.0), 1.0)
        self.assertEqual(e.update(0.0), 0.5)


if __name__ == "__main__":
    unittest.main()
