#!/usr/bin/env python3
"""
Camera-free demonstration of the detection logic.

Feeds a scripted, synthetic 60-second "drive" (EAR / MAR / head-pose values at 30 FPS)
through the SAME TemporalAnalyzer -> DrowsinessScorer -> DecisionEngine -> logger chain
that main.py uses, and prints the resulting log. Useful for:
  * demonstrating the algorithm without a camera,
  * checking that config changes behave as expected,
  * producing sample output for a report.

Run from the project root:   python tools/simulate_drive.py [--alarm simulated,audio]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config                      # noqa: E402
from src.decision import DecisionEngine                 # noqa: E402
from src.drowsiness_scorer import DrowsinessScorer      # noqa: E402
from src.event_logger import EventLogger                # noqa: E402
from src.temporal_analyzer import FrameMetrics, TemporalAnalyzer  # noqa: E402

FPS = 30
OPEN_EAR, CLOSED_EAR, REST_MAR, YAWN_MAR = 0.31, 0.12, 0.05, 0.85

# (start, end, event) - scripted scenario
SCENARIO = [
    (3.0, 3.15, "blink"), (6.5, 6.65, "blink"), (9.8, 9.95, "blink"),
    (11.0, 11.6, "face_lost"),          # brief detection dropout  -> must NOT alarm
    (14.0, 17.0, "look_side"),          # looking sideways         -> must NOT alarm
    (20.0, 23.0, "yawn"),               # yawn                     -> YAWNING, no alarm
    (26.0, 26.8, "closure"),            # 0.8 s long blink         -> no alarm
    (30.0, 32.6, "closure"),            # 2.6 s eye closure        -> ALARM
    (41.0, 44.0, "head_drop"),          # head drops for 3 s       -> ALARM
    (50.0, 50.15, "blink"), (54.0, 54.15, "blink"),
]


def metrics_at(t: float, rng) -> FrameMetrics | None:
    ear, mar, pitch, yaw = OPEN_EAR, REST_MAR, 0.0, 0.0
    for start, end, kind in SCENARIO:
        if start <= t < end:
            if kind == "face_lost":
                return None
            if kind in ("blink", "closure"):
                ear = CLOSED_EAR
            elif kind == "look_side":
                yaw, ear = 45.0, 0.17        # EAR looks "closed" when the head is turned
            elif kind == "yawn":
                mar, ear = YAWN_MAR, 0.24    # eyes narrow a little while yawning
            elif kind == "head_drop":
                pitch, ear = 28.0, 0.27
    return FrameMetrics(ear + rng.normal(0, 0.008), mar + abs(rng.normal(0, 0.01)),
                        pitch + rng.normal(0, 1.0), yaw + rng.normal(0, 1.0), 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--alarm", default="simulated")
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.logging.interval = 2.0
    rng = np.random.default_rng(1)
    analyzer, scorer, engine = TemporalAnalyzer(cfg), DrowsinessScorer(cfg), DecisionEngine(cfg)
    start = datetime.now().replace(microsecond=0)
    logger = EventLogger(cfg.logging, start)
    alarm = None
    if args.alarm != "none":
        from src.alarm import AlarmManager
        alarm = AlarmManager(cfg.alarm, args.alarm.split(","))

    def wall(t):
        return datetime.fromtimestamp(start.timestamp() + t)

    for i in range(int(args.duration * FPS)):
        t = i / FPS
        st = analyzer.update(metrics_at(t, rng), t)
        for ev in st.events:
            logger.log_event(wall(t), ev)
        sc = scorer.compute(st)
        d = engine.update(st, sc, t)
        if d.alarm_changed:
            logger.log_event(wall(t), "ALARM ACTIVATED" if d.alarm_on else "ALARM DEACTIVATED - driver alert again")
            if alarm:
                alarm.on() if d.alarm_on else alarm.off()
        logger.log_sample(t, wall(t), st, sc, d.state.value, d.alarm_on)

    logger.close(f"{wall(args.duration):%H:%M:%S} | SIMULATION ENDED | alarms: {engine.alarm_count} | "
                 f"blinks: {analyzer.total_blinks} | yawns: {analyzer.total_yawns} | nods: {analyzer.total_nods}")
    if alarm:
        alarm.close()
    print(f"\nLog written to {logger.log_path}")


if __name__ == "__main__":
    main()
