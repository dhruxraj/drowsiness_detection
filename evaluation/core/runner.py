"""Run one labelled recording through the production pipeline and record a trace.

Frames are read with the production ``src.camera.Camera`` class, exactly as
``python main.py --source <video>`` does (video files use the video's own clock).

Timing: ``processing_ms`` covers the pipeline work for one frame (pre-processing,
landmarks, EAR/MAR/pose, temporal analysis, scoring, decision) and excludes frame
acquisition/decoding, which is measured separately as ``acquire_ms``. Neither includes
the dashboard, alarm hardware or logging, which are not part of the evaluation run.
"""
from __future__ import annotations

import time
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from .pipeline import DecisionLogic, DecisionOutput
from .trace import (FAILED_OPEN, FAILED_PROCESSING, NO_FACE, NO_FRAMES, OK, STALLED, FrameRecord,
                    RecordingRun)


class StallError(RuntimeError):
    pass


class CameraFrameSource:
    """Iterates over a video file with the production ``Camera`` class."""

    def __init__(self, cfg, video_path, read_timeout_s: float = 1.0, stall_timeout_s: float = 15.0,
                 clock: Callable[[], float] = time.monotonic):
        from src.camera import Camera

        cfg.camera.source = str(video_path)  # same override main.py applies for --source
        self.camera = Camera(cfg.camera).start()
        if not getattr(self.camera, "is_file", True):
            self.camera.stop()
            raise RuntimeError(f"{video_path} was not opened as a video file")
        self.read_timeout_s = read_timeout_s
        self.stall_timeout_s = stall_timeout_s
        self._clock = clock

    def frames(self) -> Iterator[Tuple[object, float]]:
        waiting_since: Optional[float] = None
        while True:
            frame, ts = self.camera.read(timeout=self.read_timeout_s)
            if frame is None:
                if self.camera.finished:
                    return
                now = self._clock()
                if waiting_since is None:
                    waiting_since = now
                elif now - waiting_since > self.stall_timeout_s:
                    raise StallError(f"no frame for {self.stall_timeout_s}s and no end-of-file")
                continue
            waiting_since = None
            yield frame, ts

    def close(self) -> None:
        self.camera.stop()


def run_recording(recording_id: str, pipeline_factory: Callable[[], object],
                  source_factory: Callable[[], object],
                  clock: Callable[[], float] = time.perf_counter, frame_step: int = 1) -> RecordingRun:
    """Process one recording. Never raises for per-recording failures; the failure is
    recorded in ``RecordingRun.status`` so a batch run can continue.

    ``frame_step`` > 1 processes only every N-th decoded frame (timestamps stay on the video
    clock). It emulates a lower processing frame rate to check that the time-based temporal
    analysis behaves the same; such runs are never official final results."""
    if int(frame_step) < 1:
        raise ValueError("frame_step must be >= 1")
    run = RecordingRun(recording_id=recording_id, status=OK, frame_step=int(frame_step))
    pipeline = source = None
    t_setup = clock()
    try:
        pipeline = pipeline_factory()
        source = source_factory()
    except Exception as exc:  # noqa: BLE001 - reported as a recording failure
        run.status, run.error = FAILED_OPEN, f"{type(exc).__name__}: {exc}"
        _close(pipeline, source)
        return run
    run.startup_s = clock() - t_setup

    t0 = None
    decoded = 0
    frames_iter = iter(source.frames())
    try:
        while True:
            a0 = clock()
            try:
                frame, ts = next(frames_iter)
            except StopIteration:
                break
            a1 = clock()
            decoded += 1
            if t0 is None:
                t0 = ts
            if (decoded - 1) % run.frame_step:
                continue  # frame-rate variation: skipped frame (decoded, not processed)
            t = ts - t0
            m = pipeline.measure(frame, t)
            d: DecisionOutput = pipeline.decide(m.metrics, t)
            p1 = clock()
            if d.calibration_message:
                run.calibration_message = d.calibration_message
            run.frames.append(FrameRecord(
                frame_idx=len(run.frames), t_s=float(t), face_detected=m.face_detected,
                ear=_f(m.ear), mar=_f(m.mar), pitch=_f(m.pitch), yaw=_f(m.yaw), roll=_f(m.roll),
                measurement_valid=m.valid, invalid_reason=m.invalid_reason,
                calibrating=d.calibrating, state=d.state, alarm_on=d.alarm_on, score=d.score,
                events=list(d.events), processing_ms=(p1 - a1) * 1000.0, acquire_ms=(a1 - a0) * 1000.0,
            ))
    except StallError as exc:
        run.status, run.error = STALLED, str(exc)
    except Exception as exc:  # noqa: BLE001
        run.status, run.error = FAILED_PROCESSING, f"{type(exc).__name__}: {exc} (frame {len(run.frames)})"
    finally:
        _close(pipeline, source)

    if run.status == OK:
        if not run.frames:
            run.status = NO_FRAMES
        elif not any(f.face_detected for f in run.frames):
            run.status = NO_FACE
    return run


def _f(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _close(pipeline, source) -> None:
    for obj in (source, pipeline):
        if obj is not None:
            try:
                obj.close()
            except Exception:  # noqa: BLE001 - closing must not mask the real result
                pass


# ---------------------------------------------------------------------------- replay

def replay_decisions(frames: Sequence[FrameRecord], cfg, metrics_factory: Callable,
                     use_calibration: bool = True, logic_factory: Optional[Callable] = None
                     ) -> List[FrameRecord]:
    """Feed recorded per-frame measurements through the production calibration / temporal
    analysis / scoring / decision logic with configuration ``cfg``.

    Valid only for parameters that do not influence the measurements themselves (see
    ``eval_config.REPLAY_UNSAFE_PREFIXES``). Used by ``tune.py`` on TUNING data only.
    """
    logic = logic_factory(cfg) if logic_factory else DecisionLogic(cfg, use_calibration)
    out: List[FrameRecord] = []
    for f in frames:
        metrics = metrics_factory(f.ear, f.mar, f.pitch, f.yaw, f.roll) if f.face_detected else None
        d = logic.step(metrics, f.t_s)
        out.append(FrameRecord(
            frame_idx=f.frame_idx, t_s=f.t_s, face_detected=f.face_detected, ear=f.ear, mar=f.mar,
            pitch=f.pitch, yaw=f.yaw, roll=f.roll, measurement_valid=f.measurement_valid,
            invalid_reason=f.invalid_reason, calibrating=d.calibrating, state=d.state,
            alarm_on=d.alarm_on, score=d.score, events=list(d.events),
            processing_ms=f.processing_ms, acquire_ms=f.acquire_ms))
    return out
