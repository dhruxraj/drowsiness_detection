#!/usr/bin/env python3
"""
Driver Drowsiness Detection and Alarm System - main application.

Pipeline per frame:
  Camera -> pre-processing -> MediaPipe face landmarks -> EAR / MAR / head pose
         -> temporal analysis -> drowsiness score -> decision -> alarm + dashboard + log

Usage examples:
  python main.py                          # default webcam, config.yaml
  python main.py --source 1               # second camera
  python main.py --source test_video.mp4  # recorded video (timing uses the video clock)
  python main.py --no-display             # headless (e.g. Raspberry Pi without screen)
  python main.py --alarm simulated,gpio   # choose alarm outputs

SAFETY: this is a driver-assistance PROTOTYPE, not a certified automotive safety system.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from src.alarm import AlarmManager
from src.calibration import Calibrator
from src.camera import Camera, CameraError
from src.config import ConfigError, load_config
from src.decision import DecisionEngine
from src.drowsiness_scorer import DrowsinessScorer
from src.event_logger import EventLogger
from src.head_pose import HeadPoseEstimator
from src.landmark_detector import FaceLandmarkDetector, LandmarkDetectorError
from src.metrics import compute_ear, compute_mar
from src.preprocessing import Preprocessor
from src.temporal_analyzer import FrameMetrics, TemporalAnalyzer

SAFETY_NOTICE = """
======================================================================
 DRIVER DROWSINESS DETECTION - PROTOTYPE
 This is a driver-assistance prototype for education/demonstration.
 It is NOT a certified automotive safety system and must NOT be relied
 upon as the sole safety mechanism in a real vehicle. Never test it
 while actually driving - use a parked car or a desk setup.
======================================================================
"""


def parse_args():
    p = argparse.ArgumentParser(description="Real-time driver drowsiness detection")
    p.add_argument("--config", default="config.yaml", help="path to configuration file")
    p.add_argument("--source", help="camera index or video file (overrides config)")
    p.add_argument("--no-display", action="store_true", help="run without the dashboard window")
    p.add_argument("--no-calibration", action="store_true", help="skip start-up calibration")
    p.add_argument("--alarm", help="comma-separated alarm backends, e.g. simulated,audio,gpio,serial")
    return p.parse_args()


class FPSCounter:
    def __init__(self):
        self._last = None
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last is not None:
            inst = 1.0 / max(now - self._last, 1e-6)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last = now
        return self.fps


def main() -> int:
    args = parse_args()
    print(SAFETY_NOTICE)
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ConfigError) as exc:
        print(exc)
        return 1
    if args.source is not None:
        cfg.camera.source = args.source
    show = cfg.display.enabled and not args.no_display
    alarm_names = [a.strip() for a in args.alarm.split(",")] if args.alarm else None

    # ---------------- build the pipeline
    try:
        camera = Camera(cfg.camera).start()
        detector = FaceLandmarkDetector(cfg.face_mesh)
    except (CameraError, LandmarkDetectorError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"[main] landmark engine: {detector.engine}; source: {camera.source!r}")
    pre = Preprocessor(cfg.preprocessing)
    head = HeadPoseEstimator(invert_pitch=cfg.head.invert_pitch)
    analyzer = TemporalAnalyzer(cfg)
    scorer = DrowsinessScorer(cfg)
    engine = DecisionEngine(cfg)
    alarm = AlarmManager(cfg.alarm, alarm_names)
    session_start = datetime.now()
    logger = EventLogger(cfg.logging, session_start)
    calibrator = Calibrator(cfg) if (cfg.calibration.enabled and not args.no_calibration) else None
    dashboard = None
    if show:
        from src.dashboard import Dashboard  # imported only when a display is used
        dashboard = Dashboard(cfg.display, mirror=cfg.camera.mirror)
    fps = FPSCounter()
    t0 = None
    t = 0.0
    last_console = 0.0

    def wall_time(ts: float) -> datetime:
        # live cameras: real clock; video files: session start + video time
        if camera.is_file:
            return datetime.fromtimestamp(session_start.timestamp() + ts)
        return datetime.now()

    logger.log_event(wall_time(0), "SESSION STARTED")
    if calibrator:
        logger.log_event(wall_time(0), "CALIBRATION STARTED - look straight ahead, eyes open, mouth closed")

    try:
        while True:
            frame, ts = camera.read(timeout=1.0)
            if frame is None:
                if camera.finished:
                    print("[main] video/camera finished")
                    break
                continue
            if t0 is None:
                t0 = ts
            t = ts - t0

            # ---------------- per-frame measurements
            proc = pre.process(frame)
            pts = detector.detect(proc, t)
            metrics = None
            if pts is not None:
                pose = head.estimate(pts, frame.shape) if cfg.head.enabled else None
                pitch, yaw, roll = pose if pose else (None, None, None)
                metrics = FrameMetrics(compute_ear(pts), compute_mar(pts), pitch, yaw, roll)
            else:
                head.reset()

            st = score = None
            calib_progress = None
            # ---------------- calibration phase
            if calibrator is not None and not calibrator.done:
                calibrator.add(metrics, t)
                calib_progress = calibrator.progress(t)
                decision = engine.calibrating()
                if calibrator.done:
                    res = calibrator.result()
                    analyzer.apply_calibration(res.ear_threshold, res.mar_threshold, res.pitch0, res.yaw0)
                    engine.reset()
                    logger.log_event(wall_time(t), res.message)
            # ---------------- detection phase
            else:
                st = analyzer.update(metrics, t)
                for ev in st.events:
                    logger.log_event(wall_time(t), ev)
                score = scorer.compute(st)
                decision = engine.update(st, score, t)
                if decision.alarm_changed:
                    if decision.alarm_on:
                        alarm.on()
                        logger.log_event(wall_time(t), "ALARM ACTIVATED")
                    else:
                        alarm.off()
                        logger.log_event(wall_time(t), "ALARM DEACTIVATED - driver alert again")
                logger.log_sample(t, wall_time(t), st, score, decision.state.value, decision.alarm_on)

            fps.tick()

            # ---------------- output
            if dashboard is not None:
                thresholds = {"ear": analyzer.ear_threshold, "mar": analyzer.mar_threshold,
                              "score": cfg.scoring.threshold}
                img = dashboard.render(proc, pts, st, score, decision, fps.fps, thresholds, calib_progress)
                key = dashboard.show(img)
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    alarm.off()
                    calibrator = calibrator or Calibrator(cfg)
                    calibrator.start()
                    analyzer.reset()
                    engine.reset()
                    logger.log_event(wall_time(t), "RE-CALIBRATION STARTED")
                elif key == ord("t"):
                    logger.log_event(wall_time(t), "ALARM TEST")
                    alarm.test(1.5)
                elif key == ord("r"):
                    analyzer.reset()
                    engine.reset()
                    alarm.off()
                    logger.log_event(wall_time(t), "STATISTICS RESET")
            elif not cfg.logging.console and time.monotonic() - last_console > 1.0:
                last_console = time.monotonic()
                print(f"\r{decision.state.value:<13} score {score.total if score else 0:5.1f} "
                      f"fps {fps.fps:4.1f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[main] interrupted by user")
    finally:
        alarm.close()
        camera.stop()
        detector.close()
        if dashboard is not None:
            dashboard.close()
        summary = (f"{wall_time(t):%H:%M:%S} | SESSION ENDED | alarms: {engine.alarm_count} | "
                   f"blinks: {analyzer.total_blinks} | yawns: {analyzer.total_yawns} | "
                   f"nods: {analyzer.total_nods}")
        logger.close(summary)
        print(f"[main] log saved to {logger.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
