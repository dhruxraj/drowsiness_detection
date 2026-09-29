"""UNIT-TEST FIXTURES - synthetic, NOT real data.

Everything in this module is invented solely to check that the evaluation code in
``evaluation/core`` computes what it is documented to compute. None of these subjects,
recordings, labels, traces or timings exist, and nothing derived from them is an
experimental result. Real recordings and labels live only in ``evaluation/labels`` and the
git-ignored ``evaluation/dataset`` folders (see evaluation/dataset/README.md).
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from evaluation.core import eval_config
from evaluation.core.dataset import LABEL_COLUMNS, RECORDING_COLUMNS, LabelEvent, Recording
from evaluation.core.trace import FrameRecord

FIXTURE_NOTE = "UNIT-TEST FIXTURE - synthetic, not real data"

Interval = Tuple[float, float]


def eval_cfg(**overrides) -> Dict:
    """Default evaluation protocol (DEFAULTS in eval_config) with optional overrides."""
    return eval_config.from_dict(overrides)


def recording(recording_id: str = "FIXTURE_R01", subject_id: str = "FIXTURE_S01", *,
              split: str = "final", duration_s: float = 400.0, source_fps: float = 4.0,
              calibration_end_s: float = 10.0, lighting: str = "normal_indoor",
              glasses: str = "none", head_position: str = "frontal",
              camera_distance: str = "medium", camera_model: str = "fixture-cam",
              sha256: str = "") -> Recording:
    return Recording(
        recording_id=recording_id, subject_id=subject_id, split=split,
        file=f"{recording_id}.mp4", sha256=sha256, duration_s=duration_s, source_fps=source_fps,
        width=640, height=480, camera_model=camera_model, lighting=lighting, glasses=glasses,
        head_position=head_position, camera_distance=camera_distance, camera_distance_cm=None,
        calibration_end_s=calibration_end_s, consent="yes", notes=FIXTURE_NOTE)


def event(event_id: str, event_type: str, start_s: float, end_s: float, alarm_expected: str,
          recording_id: str = "FIXTURE_R01") -> LabelEvent:
    return LabelEvent(recording_id, event_id, event_type, start_s, end_s, alarm_expected,
                      "fixture", FIXTURE_NOTE)


def _inside(t: float, intervals: Iterable[Interval]) -> bool:
    return any(a <= t < b for a, b in intervals)


def make_trace(duration_s: float = 400.0, fps: float = 4.0, calibration_end_s: float = 10.0, *,
               alarms: Sequence[Interval] = (), face_missing: Sequence[Interval] = (),
               invalid: Sequence[Interval] = (), processing_ms: float = 20.0,
               acquire_ms: float = 5.0, events: Optional[Dict[float, List[str]]] = None
               ) -> List[FrameRecord]:
    """A synthetic per-frame trace. With fps=4 all timestamps are exact binary fractions,
    so interval boundaries in the tests are unambiguous."""
    frames: List[FrameRecord] = []
    n = int(round(duration_s * fps))
    for i in range(n):
        t = i / fps
        face = not _inside(t, face_missing)
        valid = face and not _inside(t, invalid)
        frames.append(FrameRecord(
            frame_idx=i, t_s=t, face_detected=face,
            ear=(0.3 if valid else (float("nan") if face else None)),
            mar=(0.05 if face else None), pitch=(0.0 if face else None),
            yaw=(0.0 if face else None), roll=(0.0 if face else None),
            measurement_valid=valid,
            invalid_reason="" if valid else ("face_not_detected" if not face else "ear_invalid"),
            calibrating=t < calibration_end_s, state="FIXTURE", alarm_on=_inside(t, alarms),
            score=None if t < calibration_end_s else 0.0,
            events=list((events or {}).get(t, [])),
            processing_ms=processing_ms, acquire_ms=acquire_ms))
    return frames


# ------------------------------------------------------------------ CSV / config files

def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return path


def manifest_row(recording_id: str, subject_id: str, split: str, **kw) -> Dict[str, object]:
    row = {
        "recording_id": recording_id, "subject_id": subject_id, "split": split,
        "file": f"{recording_id}.mp4", "sha256": "", "duration_s": 120, "source_fps": 30,
        "resolution": "640x480", "camera_model": "fixture-cam", "lighting": "normal_indoor",
        "glasses": "none", "head_position": "frontal", "camera_distance": "medium",
        "camera_distance_cm": 60, "calibration_end_s": 8, "consent": "yes",
        "recorded_on": "2000-01-01", "annotator": "fixture", "notes": FIXTURE_NOTE,
    }
    row.update(kw)
    return row


def label_row(recording_id: str, event_id: str, event_type: str, start_s, end_s,
              alarm_expected: str, **kw) -> Dict[str, object]:
    row = {"recording_id": recording_id, "event_id": event_id, "event_type": event_type,
           "start_s": start_s, "end_s": end_s, "alarm_expected": alarm_expected,
           "annotator": "fixture", "notes": FIXTURE_NOTE}
    row.update(kw)
    return row


FIXTURE_SYSTEM_CONFIG = {
    # Minimal stand-in for config.yaml, used only where tests need a detection config
    # dictionary (fingerprints, face-loss classes). Not the project's configuration.
    "eyes": {"ear_threshold": 0.21, "closure_duration_threshold": 1.8},
    "mouth": {"mar_threshold": 0.6},
    "head": {"enabled": True, "pitch_down_threshold": 18},
    "face_loss": {"grace_period": 1.0, "warn_after": 5.0},
    "calibration": {"enabled": True, "ear_ratio": 0.75},
    "scoring": {"threshold": 100},
    "alarm": {"min_duration": 3.0, "release_time": 1.5},
}


class FixtureWorkspace:
    """A temporary directory with its own evaluation config, manifests and labels.

    Nothing here touches the repository's real evaluation/labels or evaluation/results.
    """

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="eval-fixture-")
        self.root = Path(self._tmp.name)
        self.system_config = self.root / "config.yaml"
        self.system_config.write_text(yaml.safe_dump(FIXTURE_SYSTEM_CONFIG), encoding="utf-8")
        paths = {
            "system_config": str(self.system_config),
            "frozen_thresholds": str(self.root / "frozen_thresholds.yaml"),
            "results_dir": str(self.root / "results"),
            "runs_dir": str(self.root / "runs"),
            "splits": {
                split: {
                    "dataset_dir": str(self.root / "dataset" / split),
                    "recordings": str(self.root / f"{split}_recordings.csv"),
                    "labels": str(self.root / f"{split}_labels.csv"),
                } for split in ("tuning", "final")
            },
        }
        self.config_path = self.root / "evaluation_config.yaml"
        self.config_path.write_text(yaml.safe_dump({"paths": paths}), encoding="utf-8")
        (self.root / "results").mkdir()
        for split in ("tuning", "final"):
            (self.root / "dataset" / split).mkdir(parents=True)
            self.write_split(split, [], [])

    @property
    def cfg(self) -> Dict:
        return eval_config.load(self.config_path)

    def write_split(self, split: str, recordings: Sequence[Dict], labels: Sequence[Dict]) -> None:
        write_csv(self.root / f"{split}_recordings.csv", RECORDING_COLUMNS, recordings)
        write_csv(self.root / f"{split}_labels.csv", LABEL_COLUMNS, labels)

    def touch_videos(self, split: str, recording_ids: Sequence[str], content: bytes = b"") -> None:
        for rid in recording_ids:
            (self.root / "dataset" / split / f"{rid}.mp4").write_bytes(content or rid.encode())

    def cleanup(self) -> None:
        self._tmp.cleanup()
