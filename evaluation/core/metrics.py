"""Metric computation from per-frame traces + ground-truth labels.

All definitions are documented in evaluation/README.md ("Metric definitions"). Units:
time in seconds (``*_s``), latency in milliseconds (``*_ms``), rates in percent, frame
rates in frames per second.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import intervals as iv
from .dataset import LabelEvent, Recording
from .matching import MatchResult, extract_episodes, match_alarms
from .trace import EVALUATED_STATUSES, FrameRecord, RecordingRun

# Face-loss episode classes (thresholds come from config.yaml -> face_loss)
SHORT_DROPOUT = "short_dropout"      # < face_loss.grace_period: ignored by the analyzer
TEMPORARY = "temporary"              # < face_loss.warn_after
EXTENDED = "extended"                # >= face_loss.warn_after: "DRIVER NOT VISIBLE"

UNLABELLED = "unlabelled (normal)"


@dataclass
class LossEpisode:
    start_s: float
    duration_s: float
    kind: str
    expected: bool  # overlaps a labelled face_loss period


@dataclass
class RecordingEvaluation:
    recording: Recording
    status: str
    included: bool
    error: str = ""
    positives: List[LabelEvent] = field(default_factory=list)
    labels: List[LabelEvent] = field(default_factory=list)
    eval_start_s: float = 0.0
    eval_end_s: float = 0.0
    evaluated_s: float = 0.0
    non_event_s: float = 0.0
    match: Optional[MatchResult] = None
    fa_contexts: List[str] = field(default_factory=list)
    event_face_availability: Dict[str, float] = field(default_factory=dict)
    frames_all: int = 0
    frames_eval: int = 0
    frames_face_missing: int = 0
    frames_invalid: int = 0
    frames_unavailable_expected: int = 0
    frames_unavailable_unexpected: int = 0
    time_eval_s: float = 0.0
    time_face_missing_s: float = 0.0
    time_invalid_s: float = 0.0
    time_unavailable_unexpected_s: float = 0.0
    invalid_reasons: Dict[str, int] = field(default_factory=dict)
    loss_episodes: List[LossEpisode] = field(default_factory=list)
    processing_ms: List[float] = field(default_factory=list)
    acquire_ms: List[float] = field(default_factory=list)
    measured_source_fps: Optional[float] = None
    frame_shortfall_ratio: Optional[float] = None
    system_calibration_end_s: Optional[float] = None
    indicator: Dict[str, List[int]] = field(default_factory=dict)  # type -> [detected, total]
    warnings: List[str] = field(default_factory=list)


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p90": _percentile(values, 90),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _frame_durations(frames: Sequence[FrameRecord], nominal: float) -> List[float]:
    dts = [b.t_s - a.t_s for a, b in zip(frames, frames[1:])]
    dts = [d if d > 0 else 0.0 for d in dts]
    dts.append(nominal)
    return dts


def evaluate_recording(rec: Recording, labels: Sequence[LabelEvent], run: RecordingRun,
                       eval_cfg: Dict, face_loss_cfg: Dict) -> RecordingEvaluation:
    positives = [e for e in labels if e.is_positive]
    ev = RecordingEvaluation(recording=rec, status=run.status, error=run.error,
                             included=run.status in EVALUATED_STATUSES,
                             positives=positives, labels=list(labels))
    if ev.included and not run.frames:
        ev.included, ev.status = False, "no_frames"
    if not ev.included:
        return ev

    pre = float(eval_cfg["matching"]["pre_tolerance_s"])
    post = float(eval_cfg["matching"]["post_tolerance_s"])
    step = max(1, int(getattr(run, "frame_step", 1) or 1))
    nominal = step / rec.source_fps  # nominal spacing of processed frames
    frames = run.frames
    ev.frames_all = len(frames)

    # ---- evaluation window: after the protocol calibration period, until the last frame
    ev.eval_start_s = rec.calibration_end_s
    ev.eval_end_s = max(ev.eval_start_s, min(rec.duration_s, frames[-1].t_s + nominal))
    window = [(ev.eval_start_s, ev.eval_end_s)]
    pos_w = [iv.expand(e.interval, pre, post) for e in positives]
    ign_w = [iv.expand(e.interval, pre, post) for e in labels if e.is_ignore]
    ev.evaluated_s = iv.total_length(iv.subtract(window, ign_w))
    ev.non_event_s = iv.total_length(iv.subtract(window, pos_w + ign_w))

    calib_done = [f.t_s for f in frames if not f.calibrating]
    ev.system_calibration_end_s = calib_done[0] if calib_done else None
    if ev.system_calibration_end_s is None:
        ev.warnings.append(f"{rec.recording_id}: the system never left calibration")
    elif ev.system_calibration_end_s > ev.eval_start_s + nominal:
        ev.warnings.append(
            f"{rec.recording_id}: system calibration ended at {ev.system_calibration_end_s:.2f}s, "
            f"after the labelled calibration_end_s={ev.eval_start_s:.2f}s")

    # ---- alarms vs ground truth
    ev.match = match_alarms(labels, extract_episodes(frames), ev.eval_start_s, pre, post)
    context_labels = [e for e in labels if not e.is_positive and not e.is_ignore]
    for fa in ev.match.false_alarms:
        ctx = [e.event_type for e in context_labels if e.start_s <= fa.onset_s <= e.end_s]
        ev.fa_contexts.append(ctx[0] if ctx else UNLABELLED)

    # ---- measurement availability inside the evaluation window
    dts = _frame_durations(frames, nominal)
    face_loss_labels = [e.interval for e in labels if e.event_type == "face_loss"]
    eval_idx = [i for i, f in enumerate(frames) if ev.eval_start_s <= f.t_s < ev.eval_end_s]
    ev.frames_eval = len(eval_idx)
    for i in eval_idx:
        f, dt = frames[i], dts[i]
        ev.time_eval_s += dt
        unavailable = False
        if not f.face_detected:
            ev.frames_face_missing += 1
            ev.time_face_missing_s += dt
            unavailable = True
        elif not f.measurement_valid:
            ev.frames_invalid += 1
            ev.time_invalid_s += dt
            unavailable = True
            for reason in filter(None, f.invalid_reason.split(";")):
                ev.invalid_reasons[reason] = ev.invalid_reasons.get(reason, 0) + 1
        if unavailable:
            if iv.contains(face_loss_labels, f.t_s):
                ev.frames_unavailable_expected += 1
            else:
                ev.frames_unavailable_unexpected += 1
                ev.time_unavailable_unexpected_s += dt

    grace = float(face_loss_cfg.get("grace_period", 1.0))
    warn_after = float(face_loss_cfg.get("warn_after", 5.0))
    run_start: Optional[float] = None
    for i in eval_idx + [None]:
        f = frames[i] if i is not None else None
        missing = f is not None and not f.face_detected
        if missing and run_start is None:
            run_start = f.t_s
        elif not missing and run_start is not None:
            end = f.t_s if f is not None else ev.eval_end_s
            dur = end - run_start
            kind = SHORT_DROPOUT if dur < grace else (TEMPORARY if dur < warn_after else EXTENDED)
            expected = any(s < end and run_start < e for s, e in face_loss_labels)
            ev.loss_episodes.append(LossEpisode(run_start, dur, kind, expected))
            run_start = None

    for e in positives:
        inside = [f for f in frames if e.start_s <= f.t_s <= e.end_s]
        ok = sum(1 for f in inside if f.face_detected and f.measurement_valid)
        ev.event_face_availability[e.event_id] = ok / len(inside) if inside else 0.0

    # ---- performance
    ev.processing_ms = [f.processing_ms for f in frames]
    ev.acquire_ms = [f.acquire_ms for f in frames]
    if len(frames) > 1 and frames[-1].t_s > frames[0].t_s:
        ev.measured_source_fps = (len(frames) - 1) / (frames[-1].t_s - frames[0].t_s)
    expected_frames = rec.duration_s * rec.source_fps / step
    if expected_frames > 0:
        ev.frame_shortfall_ratio = max(0.0, 1.0 - len(frames) / expected_frames)
        limit = float(eval_cfg["recording_checks"]["max_frame_shortfall_ratio"])
        if ev.frame_shortfall_ratio > limit:
            ev.warnings.append(
                f"{rec.recording_id}: processed {len(frames)} frames but the manifest implies "
                f"~{expected_frames:.0f} ({ev.frame_shortfall_ratio:.1%} fewer) - frames were dropped "
                "or duration_s/source_fps in the manifest are wrong")

    # ---- supplementary: did the analyzer log the matching indicator event?
    event_frames = [f for f in frames if f.events]
    for etype, patterns in (eval_cfg.get("indicator_events") or {}).items():
        regs = [re.compile(p) for p in patterns]
        of_type = [e for e in labels if e.event_type == etype and e.start_s >= ev.eval_start_s]
        hit = 0
        for e in of_type:
            lo, hi = e.start_s - pre, e.end_s + post
            if any(lo <= f.t_s <= hi and any(r.search(msg) for r in regs for msg in f.events)
                   for f in event_frames):
                hit += 1
        ev.indicator[etype] = [hit, len(of_type)]
    return ev


def _ratio(num: float, den: float, scale: float = 1.0) -> Optional[float]:
    return (num / den) * scale if den > 0 else None


def aggregate(evals: Sequence[RecordingEvaluation]) -> Dict:
    """Pool per-recording evaluations into one summary dictionary."""
    inc = [e for e in evals if e.included]
    failed = [e for e in evals if not e.included]
    s: Dict = {
        "measured": bool(inc),
        "recordings_total": len(evals),
        "recordings_evaluated": len(inc),
        "recordings_failed": len(failed),
        "recordings_no_face": sum(1 for e in inc if e.status == "no_face_detected"),
        "failed_recordings": {e.recording.recording_id: e.status for e in failed},
        "subjects": len({e.recording.subject_id for e in evals}),
        "subjects_evaluated": len({e.recording.subject_id for e in inc}),
        "recorded_duration_s": sum(e.recording.duration_s for e in evals),
        "evaluated_s": sum(e.evaluated_s for e in inc),
        "non_event_s": sum(e.non_event_s for e in inc),
    }
    detections = [d for e in inc for d in e.match.detections]
    missed = [m for e in inc for m in e.match.missed]
    s["positive_events"] = sum(len(e.positives) for e in inc)
    s["positive_events_not_evaluated"] = sum(len(e.positives) for e in failed)
    s["detected_events"] = len(detections)
    s["missed_events"] = len(missed) if inc else None
    s["detection_rate_pct"] = _ratio(len(detections), s["positive_events"], 100.0)
    s["missed_event_ids"] = [f"{m.recording_id}/{m.event_id}" for m in missed]
    s["missed_while_face_unavailable"] = sum(
        1 for e in inc for m in e.match.missed if e.event_face_availability.get(m.event_id, 0) < 0.5)
    fa = sum(len(e.match.false_alarms) for e in inc)
    s["false_alarms"] = fa if inc else None
    s["false_alarms_per_hour"] = _ratio(fa, s["non_event_s"], 3600.0) if inc else None
    s["false_alarms_per_evaluated_hour"] = _ratio(fa, s["evaluated_s"], 3600.0) if inc else None
    s["duplicate_alarms"] = sum(len(e.match.duplicates) for e in inc)
    s["ignored_alarms"] = sum(len(e.match.ignored) for e in inc)
    s["alarms_during_calibration"] = sum(len(e.match.before_window) for e in inc)
    s["latency_ms"] = summary_stats([d.latency_ms for d in detections])
    s["release_delay_ms"] = summary_stats([d.release_delay_ms for d in detections
                                           if d.release_delay_ms is not None])

    by_type: Dict[str, Dict] = {}
    for e in inc:
        for p in e.positives:
            t = by_type.setdefault(p.event_type, {"events": 0, "detected": 0, "latencies": []})
            t["events"] += 1
        for d in e.match.detections:
            t = by_type[d.event.event_type]
            t["detected"] += 1
            t["latencies"].append(d.latency_ms)
    s["by_event_type"] = {
        k: {"events": v["events"], "detected": v["detected"], "missed": v["events"] - v["detected"],
            "latency_median_ms": statistics.median(v["latencies"]) if v["latencies"] else None}
        for k, v in sorted(by_type.items())}
    ctx: Dict[str, int] = {}
    for e in inc:
        for c in e.fa_contexts:
            ctx[c] = ctx.get(c, 0) + 1
    s["false_alarms_by_context"] = dict(sorted(ctx.items()))

    # ---- measurement availability
    frames_eval = sum(e.frames_eval for e in inc)
    time_eval = sum(e.time_eval_s for e in inc)
    fm = sum(e.frames_face_missing for e in inc)
    fi = sum(e.frames_invalid for e in inc)
    fu = sum(e.frames_unavailable_unexpected for e in inc)
    s["frames_evaluated"] = frames_eval
    s["face_missing_rate_pct"] = _ratio(fm, frames_eval, 100.0)
    s["invalid_measurement_rate_pct"] = _ratio(fi, frames_eval, 100.0)
    s["measurement_unavailable_rate_pct"] = _ratio(fm + fi, frames_eval, 100.0)
    s["unexpected_unavailable_rate_pct"] = _ratio(fu, frames_eval, 100.0)
    s["face_missing_time_pct"] = _ratio(sum(e.time_face_missing_s for e in inc), time_eval, 100.0)
    s["measurement_unavailable_time_pct"] = _ratio(
        sum(e.time_face_missing_s + e.time_invalid_s for e in inc), time_eval, 100.0)
    reasons: Dict[str, int] = {}
    for e in inc:
        for r, n in e.invalid_reasons.items():
            reasons[r] = reasons.get(r, 0) + n
    s["invalid_reasons"] = dict(sorted(reasons.items()))
    episodes = [ep for e in inc for ep in e.loss_episodes]
    s["face_loss_episodes"] = {k: sum(1 for ep in episodes if ep.kind == k)
                               for k in (SHORT_DROPOUT, TEMPORARY, EXTENDED)}
    s["face_loss_episodes_unexpected"] = sum(1 for ep in episodes if not ep.expected)
    s["longest_face_loss_s"] = max((ep.duration_s for ep in episodes), default=None)

    # ---- processing performance (all processed frames incl. calibration; failed runs excluded)
    proc = [ms for e in inc for ms in e.processing_ms]
    acq = [ms for e in inc for ms in e.acquire_ms]
    s["frames_processed"] = len(proc)
    s["processing_fps_mean"] = _ratio(len(proc), sum(proc) / 1000.0) if proc else None
    med = statistics.median(proc) if proc else None
    s["processing_fps_median"] = 1000.0 / med if med else None
    p95 = _percentile(proc, 95)
    s["processing_fps_p5"] = 1000.0 / p95 if p95 else None
    per_rec = [len(e.processing_ms) / (sum(e.processing_ms) / 1000.0)
               for e in inc if e.processing_ms and sum(e.processing_ms) > 0]
    s["processing_fps_min_recording"] = min(per_rec) if per_rec else None
    total_wall = (sum(proc) + sum(acq)) / 1000.0
    s["end_to_end_fps"] = _ratio(len(proc), total_wall) if proc else None
    s["processing_ms"] = summary_stats(proc)
    s["source_fps_median"] = statistics.median([e.recording.source_fps for e in inc]) if inc else None
    measured = [e.measured_source_fps for e in inc if e.measured_source_fps]
    s["source_fps_measured_median"] = statistics.median(measured) if measured else None

    ind: Dict[str, List[int]] = {}
    for e in inc:
        for k, (hit, tot) in e.indicator.items():
            cur = ind.setdefault(k, [0, 0])
            cur[0] += hit
            cur[1] += tot
    s["indicator_recall"] = {k: {"detected": h, "events": t, "recall_pct": _ratio(h, t, 100.0)}
                             for k, (h, t) in sorted(ind.items())}
    s["warnings"] = [w for e in evals for w in e.warnings]
    return s


def breakdowns(evals: Sequence[RecordingEvaluation], eval_cfg: Dict) -> Dict[str, Dict[str, Dict]]:
    """Aggregate per value of each condition attribute (subject, lighting, glasses, ...)."""
    cfg = eval_cfg["breakdowns"]
    out: Dict[str, Dict[str, Dict]] = {}
    for attr in cfg["attributes"]:
        groups: Dict[str, List[RecordingEvaluation]] = {}
        for e in evals:
            groups.setdefault(e.recording.attribute(attr), []).append(e)
        out[attr] = {}
        for value, members in sorted(groups.items()):
            s = aggregate(members)
            s["insufficient_positive_events"] = s["positive_events"] < cfg["min_positive_events"]
            s["insufficient_non_event_time"] = s["non_event_s"] < cfg["min_non_event_minutes"] * 60.0
            out[attr][value] = s
    return out


def dataset_summary(recordings: Sequence[Recording], labels: Dict[str, Sequence[LabelEvent]]) -> Dict:
    """Counts describing the labelled dataset (independent of any run)."""
    s: Dict = {
        "subjects": len({r.subject_id for r in recordings}),
        "recordings": len(recordings),
        "total_duration_s": sum(r.duration_s for r in recordings),
        "conditions": {},
        "event_counts": {},
        "cameras": sorted({r.camera_model or "unspecified" for r in recordings}),
        "resolutions": sorted({r.resolution for r in recordings}),
        "source_fps": sorted({r.source_fps for r in recordings}),
        "recording_ids": sorted(r.recording_id for r in recordings),
        "subject_ids": sorted({r.subject_id for r in recordings}),
    }
    for attr in ("lighting", "glasses", "head_position", "camera_distance"):
        groups: Dict[str, Dict] = {}
        for r in recordings:
            g = groups.setdefault(r.attribute(attr), {"recordings": 0, "subjects": set(), "duration_s": 0.0})
            g["recordings"] += 1
            g["subjects"].add(r.subject_id)
            g["duration_s"] += r.duration_s
        s["conditions"][attr] = {k: {"recordings": v["recordings"], "subjects": len(v["subjects"]),
                                     "duration_s": v["duration_s"]} for k, v in sorted(groups.items())}
    for events in labels.values():
        for e in events:
            c = s["event_counts"].setdefault(e.event_type, {"count": 0, "duration_s": 0.0,
                                                            "yes": 0, "no": 0, "ignore": 0})
            c["count"] += 1
            c["duration_s"] += e.duration_s
            c[e.alarm_expected] += 1
    s["event_counts"] = dict(sorted(s["event_counts"].items()))
    return s
