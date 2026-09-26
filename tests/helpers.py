"""Helpers for feeding synthetic signals through the detection pipeline."""
from __future__ import annotations

import os

from src.config import load_config
from src.decision import DecisionEngine
from src.drowsiness_scorer import DrowsinessScorer
from src.temporal_analyzer import FrameMetrics, TemporalAnalyzer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 30
OPEN, CLOSED = 0.31, 0.10


def load_cfg():
    return load_config(os.path.join(ROOT, "config.yaml"))


class Pipeline:
    """Analyzer + scorer + decision engine driven by a synthetic clock."""

    def __init__(self, cfg=None):
        self.cfg = cfg or load_cfg()
        self.an = TemporalAnalyzer(self.cfg)
        self.sc = DrowsinessScorer(self.cfg)
        self.de = DecisionEngine(self.cfg)
        self.t = 0.0
        self.alarm_times: list[float] = []
        self.states: list[str] = []
        self.last = None

    def run(self, seconds: float, ear=OPEN, mar=0.05, pitch=0.0, yaw=0.0, face=True):
        for _ in range(int(round(seconds * FPS))):
            m = FrameMetrics(ear, mar, pitch, yaw, 0.0) if face else None
            st = self.an.update(m, self.t)
            score = self.sc.compute(st)
            d = self.de.update(st, score, self.t)
            if d.alarm_changed and d.alarm_on:
                self.alarm_times.append(self.t)
            self.states.append(d.state.value)
            self.last = (st, score, d)
            self.t += 1.0 / FPS
        return self.last

    @property
    def alarm_on(self) -> bool:
        return self.de.alarm_on
