"""Adapter that runs the *production* detection pipeline frame by frame.

This module does not contain any detection logic of its own. It composes the real
modules from ``src/`` in exactly the order used by the per-frame loop in ``main.py``:

    Preprocessor.process -> FaceLandmarkDetector.detect -> HeadPoseEstimator.estimate
    -> compute_ear / compute_mar -> FrameMetrics
    -> Calibrator (start-up phase) | TemporalAnalyzer.update -> DrowsinessScorer.compute
       -> DecisionEngine.update

What is intentionally left out compared with ``main.py``: the dashboard window, the
physical alarm backends (the evaluation uses the decision engine's ``alarm_on`` flag,
i.e. the moment the software decides to switch the alarm on) and the session logger.

MAINTENANCE: if the per-frame loop in ``main.py`` changes, mirror the change here.
``tests/test_evaluation_pipeline_sync.py`` fails when the calls listed in
``MIRRORED_MAIN_CALLS`` disappear from ``main.py`` to make such drift visible.
"""
from __future__ import annotations

import copy
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Calls of main.py's per-frame loop that are mirrored below (checked by the drift test).
MIRRORED_MAIN_CALLS = (
    r"\.process\(frame\)",
    r"detector\.detect\(",
    r"head\.estimate\(pts, frame\.shape\)",
    r"head\.reset\(\)",
    r"compute_ear\(pts\)",
    r"compute_mar\(pts\)",
    r"FrameMetrics\(",
    r"calibrator\.add\(metrics, t\)",
    r"engine\.calibrating\(\)",
    r"analyzer\.apply_calibration\(",
    r"engine\.reset\(\)",
    r"analyzer\.update\(metrics, t\)",
    r"scorer\.compute\(st\)",
    r"engine\.update\(st, score, t\)",
)


@dataclass
class Measurement:
    face_detected: bool
    ear: Optional[float] = None
    mar: Optional[float] = None
    pitch: Optional[float] = None
    yaw: Optional[float] = None
    roll: Optional[float] = None
    valid: bool = False
    invalid_reason: str = ""
    metrics: Any = None  # the FrameMetrics object handed to the production analyzer (or None)


@dataclass
class DecisionOutput:
    calibrating: bool
    alarm_on: bool
    alarm_changed: bool
    state: str
    score: Optional[float]
    events: List[str] = field(default_factory=list)
    calibration_message: str = ""


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def assess_measurement(ear, mar, pitch, yaw, head_enabled: bool) -> Tuple[bool, str]:
    """Classify a frame's measurements as valid or name why they are not usable."""
    reasons = []
    if not _finite(ear):
        reasons.append("ear_invalid")
    if not _finite(mar):
        reasons.append("mar_invalid")
    if head_enabled and not (_finite(pitch) and _finite(yaw)):
        reasons.append("head_pose_unavailable")
    return (not reasons, ";".join(reasons))


def _state_str(decision) -> str:
    state = getattr(decision, "state", "")
    return str(getattr(state, "value", state))


def _score_total(score) -> Optional[float]:
    total = getattr(score, "total", None) if score is not None else None
    return float(total) if _finite(total) else None


class DecisionLogic:
    """Calibration + temporal analysis + scoring + decision, as in main.py.

    ``components`` exists only so unit tests can inject fakes; by default the real
    classes are imported from ``src``.
    """

    def __init__(self, cfg, use_calibration: bool = True, components: Optional[Sequence] = None):
        if components is None:
            from src.calibration import Calibrator
            from src.decision import DecisionEngine
            from src.drowsiness_scorer import DrowsinessScorer
            from src.temporal_analyzer import TemporalAnalyzer
            components = (TemporalAnalyzer, DrowsinessScorer, DecisionEngine, Calibrator)
        TemporalAnalyzer, DrowsinessScorer, DecisionEngine, Calibrator = components
        self.analyzer = TemporalAnalyzer(cfg)
        self.scorer = DrowsinessScorer(cfg)
        self.engine = DecisionEngine(cfg)
        self.calibrator = Calibrator(cfg) if (cfg.calibration.enabled and use_calibration) else None

    def step(self, metrics, t: float) -> DecisionOutput:
        # ---------------- calibration phase (main.py: "calibration phase")
        if self.calibrator is not None and not self.calibrator.done:
            self.calibrator.add(metrics, t)
            decision = self.engine.calibrating()
            message = ""
            if self.calibrator.done:
                res = self.calibrator.result()
                self.analyzer.apply_calibration(res.ear_threshold, res.mar_threshold, res.pitch0, res.yaw0)
                self.engine.reset()
                message = str(getattr(res, "message", ""))
            return DecisionOutput(calibrating=True, alarm_on=bool(getattr(decision, "alarm_on", False)),
                                  alarm_changed=False, state=_state_str(decision), score=None,
                                  calibration_message=message)
        # ---------------- detection phase (main.py: "detection phase")
        st = self.analyzer.update(metrics, t)
        events = [str(ev) for ev in (getattr(st, "events", None) or [])]
        score = self.scorer.compute(st)
        decision = self.engine.update(st, score, t)
        return DecisionOutput(calibrating=False, alarm_on=bool(decision.alarm_on),
                              alarm_changed=bool(decision.alarm_changed), state=_state_str(decision),
                              score=_score_total(score), events=events)


def make_frame_metrics(ear, mar, pitch, yaw, roll):
    from src.temporal_analyzer import FrameMetrics
    return FrameMetrics(ear, mar, pitch, yaw, roll)


class ProductionPipeline:
    """Full per-frame pipeline built from the production modules (needs MediaPipe/OpenCV)."""

    def __init__(self, cfg, use_calibration: bool = True):
        from src.head_pose import HeadPoseEstimator
        from src.landmark_detector import FaceLandmarkDetector
        from src.metrics import compute_ear, compute_mar
        from src.preprocessing import Preprocessor
        from src.temporal_analyzer import FrameMetrics

        self.cfg = cfg
        self._compute_ear, self._compute_mar, self._FrameMetrics = compute_ear, compute_mar, FrameMetrics
        self.pre = Preprocessor(cfg.preprocessing)
        self.detector = FaceLandmarkDetector(cfg.face_mesh)
        self.head = HeadPoseEstimator(invert_pitch=cfg.head.invert_pitch)
        self.logic = DecisionLogic(cfg, use_calibration)
        self.engine_name = getattr(self.detector, "engine", "")

    def measure(self, frame, t: float) -> Measurement:
        proc = self.pre.process(frame)
        pts = self.detector.detect(proc, t)
        if pts is None:
            self.head.reset()
            return Measurement(face_detected=False, invalid_reason="face_not_detected")
        pose = self.head.estimate(pts, frame.shape) if self.cfg.head.enabled else None
        pitch, yaw, roll = pose if pose else (None, None, None)
        ear, mar = self._compute_ear(pts), self._compute_mar(pts)
        valid, reason = assess_measurement(ear, mar, pitch, yaw, bool(self.cfg.head.enabled))
        return Measurement(True, ear, mar, pitch, yaw, roll, valid, reason,
                           self._FrameMetrics(ear, mar, pitch, yaw, roll))

    def decide(self, metrics, t: float) -> DecisionOutput:
        return self.logic.step(metrics, t)

    def close(self) -> None:
        self.detector.close()


# ---------------------------------------------------------------------------- config

def read_config_dict(path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def apply_overrides(data: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy of ``data`` with dotted-key overrides applied (keys must already exist)."""
    out = copy.deepcopy(data)
    for dotted, value in (overrides or {}).items():
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"config key {dotted!r} does not exist")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise KeyError(f"config key {dotted!r} does not exist")
        node[parts[-1]] = value
    return out


def load_production_config(path, overrides: Optional[Dict[str, Any]] = None):
    """Load ``config.yaml`` through the production loader (``src.config.load_config``),
    so the same validation applies. Overrides are written to a temporary file placed next
    to the original, so any path handling relative to the config file stays identical."""
    from src.config import load_config

    path = Path(path)
    if not overrides:
        return load_config(str(path))
    import yaml

    data = apply_overrides(read_config_dict(path), overrides)
    fd, tmp = tempfile.mkstemp(prefix=".eval-override-", suffix=".yaml", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
        return load_config(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
