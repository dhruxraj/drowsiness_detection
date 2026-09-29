"""Recording manifests and ground-truth labels.

Two CSV files describe each split (see ``evaluation/dataset/README.md`` for the full
specification):

* ``*_recordings.csv`` - one row per recording: subject, conditions, camera, FPS, ...
* ``*_labels.csv``     - one row per labelled event interval in a recording.

All timestamps are seconds on the *video clock* (0 = first frame of the file), which is
the same clock the production ``Camera`` class uses for video files.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Sequence, Tuple

from .eval_config import ALARM_EXPECTED_VALUES

SPLITS = ("tuning", "final")

RECORDING_COLUMNS = (
    "recording_id", "subject_id", "split", "file", "sha256", "duration_s", "source_fps",
    "resolution", "camera_model", "lighting", "glasses", "head_position", "camera_distance",
    "camera_distance_cm", "calibration_end_s", "consent", "recorded_on", "annotator", "notes",
)
REQUIRED_RECORDING_COLUMNS = (
    "recording_id", "subject_id", "split", "file", "duration_s", "source_fps", "resolution",
    "lighting", "glasses", "head_position", "camera_distance", "calibration_end_s", "consent",
)
LABEL_COLUMNS = (
    "recording_id", "event_id", "event_type", "start_s", "end_s", "alarm_expected",
    "annotator", "notes",
)
REQUIRED_LABEL_COLUMNS = (
    "recording_id", "event_id", "event_type", "start_s", "end_s", "alarm_expected",
)
CONDITION_ATTRIBUTES = ("subject_id", "lighting", "glasses", "head_position", "camera_distance")

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RES_RE = re.compile(r"^(\d+)\s*[xX]\s*(\d+)$")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class DatasetError(ValueError):
    """Raised with *all* problems found, so a whole file can be fixed in one pass."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class Recording:
    recording_id: str
    subject_id: str
    split: str
    file: str
    sha256: str
    duration_s: float
    source_fps: float
    width: int
    height: int
    camera_model: str
    lighting: str
    glasses: str
    head_position: str
    camera_distance: str
    camera_distance_cm: Optional[float]
    calibration_end_s: float
    consent: str
    recorded_on: str = ""
    annotator: str = ""
    notes: str = ""

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def attribute(self, name: str) -> str:
        return str(getattr(self, name))


@dataclass(frozen=True)
class LabelEvent:
    recording_id: str
    event_id: str
    event_type: str
    start_s: float
    end_s: float
    alarm_expected: str  # yes | no | ignore
    annotator: str = ""
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def is_positive(self) -> bool:
        return self.alarm_expected == "yes"

    @property
    def is_ignore(self) -> bool:
        return self.alarm_expected == "ignore"

    @property
    def interval(self) -> Tuple[float, float]:
        return (self.start_s, self.end_s)


def _read_rows(path: Path, required: Sequence[str], errors: List[str]):
    if not path.is_file():
        raise DatasetError([f"{path}: file not found"])
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise DatasetError([f"{path.name}: missing required column(s): {', '.join(missing)}"])
        rows = []
        for raw in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k is not None}
            if not any(row.values()):
                continue
            rows.append((reader.line_num, row))
    return rows


def _float(value: str, name: str, where: str, errors: List[str],
           minimum: Optional[float] = None, strictly_positive: bool = False) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        errors.append(f"{where}: {name} must be a number (got {value!r})")
        return None
    if not math.isfinite(x):
        errors.append(f"{where}: {name} must be finite (got {value!r})")
        return None
    if strictly_positive and x <= 0:
        errors.append(f"{where}: {name} must be > 0 (got {value!r})")
        return None
    if minimum is not None and x < minimum:
        errors.append(f"{where}: {name} must be >= {minimum} (got {value!r})")
        return None
    return x


def _check_vocab(value: str, allowed: Sequence[str], name: str, where: str,
                 errors: List[str]) -> None:
    if value not in allowed:
        errors.append(f"{where}: {name} {value!r} is not in the vocabulary {list(allowed)} "
                      "(extend evaluation_config.yaml -> vocabulary if a new condition is needed)")


def _safe_relative(file_value: str) -> bool:
    for cls in (PurePosixPath, PureWindowsPath):
        p = cls(file_value)
        if p.is_absolute() or p.drive or ".." in p.parts:
            return False
    return True


def load_recordings(path, expected_split: str, vocabulary: Dict, dataset_dir=None,
                    check_files: bool = False) -> Tuple[List[Recording], List[str]]:
    """Load and validate a recordings manifest. Returns ``(recordings, warnings)``."""
    path = Path(path)
    if expected_split not in SPLITS:
        raise ValueError(f"expected_split must be one of {SPLITS}")
    errors: List[str] = []
    warnings: List[str] = []
    rows = _read_rows(path, REQUIRED_RECORDING_COLUMNS, errors)
    recordings: List[Recording] = []
    seen = set()
    for line, row in rows:
        where = f"{path.name}:{line}"
        n_err = len(errors)
        rid = row.get("recording_id", "")
        sid = row.get("subject_id", "")
        for name, value in (("recording_id", rid), ("subject_id", sid)):
            if not _ID_RE.match(value):
                errors.append(f"{where}: {name} {value!r} must match {_ID_RE.pattern} "
                              "(use pseudonymous IDs such as S01, never names)")
        if rid.lower() in seen:
            errors.append(f"{where}: duplicate recording_id {rid!r}")
        seen.add(rid.lower())
        split = row.get("split", "")
        if split != expected_split:
            errors.append(f"{where}: split is {split!r} but this manifest is for "
                          f"{expected_split!r} recordings - tuning and final data must stay separate")
        file_value = row.get("file", "")
        if not file_value:
            errors.append(f"{where}: file is empty")
        elif not _safe_relative(file_value):
            errors.append(f"{where}: file {file_value!r} must be a relative path inside the "
                          "split's dataset directory")
        elif check_files and dataset_dir is not None and not (Path(dataset_dir) / file_value).is_file():
            errors.append(f"{where}: recording file not found: {Path(dataset_dir) / file_value}")
        sha = row.get("sha256", "")
        if sha and not _SHA_RE.match(sha):
            errors.append(f"{where}: sha256 must be 64 hex characters")
        elif not sha:
            warnings.append(f"{where}: sha256 is empty - run validate_dataset.py --write-hashes "
                            "so accidental re-use of a file across splits can be detected")
        duration = _float(row.get("duration_s", ""), "duration_s", where, errors, strictly_positive=True)
        fps = _float(row.get("source_fps", ""), "source_fps", where, errors, strictly_positive=True)
        calib = _float(row.get("calibration_end_s", ""), "calibration_end_s", where, errors, minimum=0.0)
        if duration is not None and calib is not None and calib >= duration:
            errors.append(f"{where}: calibration_end_s must be < duration_s")
        m = _RES_RE.match(row.get("resolution", ""))
        if not m:
            errors.append(f"{where}: resolution must look like 640x480")
        dist_cm = None
        if row.get("camera_distance_cm", ""):
            dist_cm = _float(row["camera_distance_cm"], "camera_distance_cm", where, errors,
                             strictly_positive=True)
        for key in ("lighting", "glasses", "head_position", "camera_distance"):
            _check_vocab(row.get(key, ""), vocabulary[key], key, where, errors)
        if row.get("consent", "").lower() != "yes":
            errors.append(f"{where}: consent must be 'yes' - recordings without documented "
                          "consent may not be used")
        if len(errors) == n_err:
            recordings.append(Recording(
                recording_id=rid, subject_id=sid, split=split, file=file_value, sha256=sha.lower(),
                duration_s=duration, source_fps=fps, width=int(m.group(1)), height=int(m.group(2)),
                camera_model=row.get("camera_model", ""), lighting=row["lighting"],
                glasses=row["glasses"], head_position=row["head_position"],
                camera_distance=row["camera_distance"], camera_distance_cm=dist_cm,
                calibration_end_s=calib, consent="yes", recorded_on=row.get("recorded_on", ""),
                annotator=row.get("annotator", ""), notes=row.get("notes", ""),
            ))
    if errors:
        raise DatasetError(errors)
    return recordings, warnings


def load_labels(path, recordings: Sequence[Recording], vocabulary: Dict,
                ground_truth: Dict) -> Tuple[Dict[str, List[LabelEvent]], List[str]]:
    """Load and validate labels for ``recordings``.

    Returns ``({recording_id: [events sorted by start]}, warnings)``; every recording has
    an entry (possibly empty = the whole recording is a non-event period).
    """
    path = Path(path)
    errors: List[str] = []
    warnings: List[str] = []
    rows = _read_rows(path, REQUIRED_LABEL_COLUMNS, errors)
    by_id = {r.recording_id: r for r in recordings}
    event_types = vocabulary["event_types"]
    labels: Dict[str, List[LabelEvent]] = {r.recording_id: [] for r in recordings}
    seen = set()
    for line, row in rows:
        where = f"{path.name}:{line}"
        n_err = len(errors)
        rid = row.get("recording_id", "")
        rec = by_id.get(rid)
        if rec is None:
            errors.append(f"{where}: recording_id {rid!r} is not in the recordings manifest of this split")
        eid = row.get("event_id", "")
        if not _ID_RE.match(eid):
            errors.append(f"{where}: event_id {eid!r} must match {_ID_RE.pattern}")
        if (rid, eid) in seen:
            errors.append(f"{where}: duplicate event_id {eid!r} in recording {rid!r}")
        seen.add((rid, eid))
        etype = row.get("event_type", "")
        if etype not in event_types:
            errors.append(f"{where}: unknown event_type {etype!r}; allowed: {sorted(event_types)}")
        expected = row.get("alarm_expected", "").lower()
        if expected not in ALARM_EXPECTED_VALUES:
            errors.append(f"{where}: alarm_expected must be one of {ALARM_EXPECTED_VALUES}")
        elif etype in event_types and expected not in event_types[etype]:
            errors.append(f"{where}: alarm_expected={expected!r} is not allowed for event_type "
                          f"{etype!r} (allowed: {event_types[etype]})")
        start = _float(row.get("start_s", ""), "start_s", where, errors, minimum=0.0)
        end = _float(row.get("end_s", ""), "end_s", where, errors, minimum=0.0)
        if start is not None and end is not None and end <= start:
            errors.append(f"{where}: end_s must be greater than start_s")
        if rec is not None and end is not None:
            slack = 1.0 / rec.source_fps
            if end > rec.duration_s + slack:
                errors.append(f"{where}: end_s {end} is beyond the recording duration {rec.duration_s}")
        if rec is not None and start is not None and expected == "yes" and start < rec.calibration_end_s:
            errors.append(f"{where}: an alarm-expected event starts inside the calibration period "
                          f"(< calibration_end_s={rec.calibration_end_s}); it cannot be evaluated")
        if len(errors) == n_err:
            labels[rid].append(LabelEvent(rid, eid, etype, start, end, expected,
                                          row.get("annotator", ""), row.get("notes", "")))

    for rid, events in labels.items():
        events.sort(key=lambda e: (e.start_s, e.end_s, e.event_id))
        positives = [e for e in events if e.is_positive]
        for a, b in zip(positives, positives[1:]):
            if b.start_s < a.end_s:
                errors.append(f"{path.name}: alarm-expected events {a.event_id!r} and {b.event_id!r} "
                              f"overlap in recording {rid!r}; merge them into one event")
        for e in events:
            _ground_truth_warnings(e, ground_truth, warnings, path.name)
        if not events:
            warnings.append(f"{path.name}: recording {rid!r} has no labels - it will be evaluated "
                            "as a non-event recording")
    if errors:
        raise DatasetError(errors)
    return labels, warnings


def _ground_truth_warnings(e: LabelEvent, gt: Dict, warnings: List[str], fname: str) -> None:
    """Flag labels that disagree with the protocol definitions (warnings, not errors)."""
    tag = f"{fname}: {e.recording_id}/{e.event_id}"
    if e.event_type == "eye_closure":
        if e.is_positive and e.duration_s < gt["prolonged_closure_min_s"]:
            warnings.append(f"{tag}: eye_closure marked alarm_expected=yes but lasts only "
                            f"{e.duration_s:.2f}s (< protocol {gt['prolonged_closure_min_s']}s)")
        if e.alarm_expected == "no" and e.duration_s > gt["brief_closure_max_s"]:
            warnings.append(f"{tag}: eye_closure of {e.duration_s:.2f}s marked alarm_expected=no; "
                            f"closures longer than {gt['brief_closure_max_s']}s but shorter than "
                            f"{gt['prolonged_closure_min_s']}s should normally be 'ignore'")
    if e.event_type == "head_drop" and e.is_positive and e.duration_s < gt["sustained_head_drop_min_s"]:
        warnings.append(f"{tag}: head_drop marked alarm_expected=yes but lasts only "
                        f"{e.duration_s:.2f}s (< protocol {gt['sustained_head_drop_min_s']}s)")
