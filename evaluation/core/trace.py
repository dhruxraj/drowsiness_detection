"""Per-frame traces produced by running a recording through the production pipeline.

One CSV per recording is written to ``<run_dir>/traces/<recording_id>.csv``. The trace
contains only derived numbers (EAR/MAR/pose, state, alarm flag, timing) - no images -
but it is still derived from biometric recordings, so run directories are git-ignored.
Metrics can be recomputed from traces without re-running MediaPipe
(``evaluation/scripts/calculate_metrics.py``).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# Recording-level run status
OK = "ok"
NO_FACE = "no_face_detected"          # processed, but the face was never found (measurement failure)
NO_FRAMES = "no_frames"               # opened, but no frame could be decoded
FAILED_OPEN = "failed_to_open"        # camera/detector could not be created for the file
STALLED = "stalled"                   # frames stopped arriving without end-of-file
FAILED_PROCESSING = "failed_during_processing"  # exception inside the pipeline

# Recordings with these statuses are included in the alarm metrics. A recording in which the
# face is never detected is a *system* failure, so its alarm-expected events count as missed.
EVALUATED_STATUSES = (OK, NO_FACE)
FAILED_STATUSES = (NO_FRAMES, FAILED_OPEN, STALLED, FAILED_PROCESSING)

TRACE_COLUMNS = (
    "frame_idx", "t_s", "face_detected", "ear", "mar", "pitch", "yaw", "roll",
    "measurement_valid", "invalid_reason", "calibrating", "state", "alarm_on", "score",
    "events", "processing_ms", "acquire_ms",
)


@dataclass
class FrameRecord:
    frame_idx: int
    t_s: float
    face_detected: bool
    ear: Optional[float]
    mar: Optional[float]
    pitch: Optional[float]
    yaw: Optional[float]
    roll: Optional[float]
    measurement_valid: bool
    invalid_reason: str
    calibrating: bool
    state: str
    alarm_on: bool
    score: Optional[float]
    events: List[str]
    processing_ms: float
    acquire_ms: float


@dataclass
class RecordingRun:
    recording_id: str
    status: str
    error: str = ""
    startup_s: float = 0.0
    calibration_message: str = ""
    frames: List[FrameRecord] = field(default_factory=list)
    frame_step: int = 1  # 1 = every frame processed; N = every N-th frame (frame-rate variation)

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("frames")
        d["frames_processed"] = len(self.frames)
        return d


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)  # shortest round-trip representation -> exact replay
    return str(value)


def _opt_float(s: str) -> Optional[float]:
    return None if s == "" else float(s)


def write_trace(path, frames: List[FrameRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(TRACE_COLUMNS)
        for f in frames:
            row = []
            for col in TRACE_COLUMNS:
                v = getattr(f, col)
                row.append(json.dumps(v) if col == "events" else _fmt(v))
            w.writerow(row)


def read_trace(path) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            frames.append(FrameRecord(
                frame_idx=int(row["frame_idx"]),
                t_s=float(row["t_s"]),
                face_detected=row["face_detected"] == "1",
                ear=_opt_float(row["ear"]),
                mar=_opt_float(row["mar"]),
                pitch=_opt_float(row["pitch"]),
                yaw=_opt_float(row["yaw"]),
                roll=_opt_float(row["roll"]),
                measurement_valid=row["measurement_valid"] == "1",
                invalid_reason=row["invalid_reason"],
                calibrating=row["calibrating"] == "1",
                state=row["state"],
                alarm_on=row["alarm_on"] == "1",
                score=_opt_float(row["score"]),
                events=json.loads(row["events"]) if row["events"] else [],
                processing_ms=float(row["processing_ms"]),
                acquire_ms=float(row["acquire_ms"]),
            ))
    return frames
