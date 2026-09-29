"""Orchestration shared by the command-line scripts in ``evaluation/scripts``."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import report as rp
from . import thresholds as th
from .metrics import RecordingEvaluation, aggregate, breakdowns, dataset_summary, evaluate_recording
from .provenance import utc_now
from .splits import SplitData
from .trace import RecordingRun, read_trace

RUN_META = "run.json"
TRACES_DIR = "traces"


def load_frozen(path) -> Optional[Dict]:
    import yaml

    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or None


def load_run(run_dir) -> Tuple[Dict, Dict[str, RecordingRun]]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / RUN_META).read_text(encoding="utf-8"))
    runs: Dict[str, RecordingRun] = {}
    for rid, rmeta in meta["recordings"].items():
        trace_path = run_dir / TRACES_DIR / f"{rid}.csv"
        frames = read_trace(trace_path) if trace_path.is_file() else []
        runs[rid] = RecordingRun(recording_id=rid, status=rmeta["status"], error=rmeta.get("error", ""),
                                 startup_s=rmeta.get("startup_s", 0.0),
                                 calibration_message=rmeta.get("calibration_message", ""), frames=frames,
                                 frame_step=int(rmeta.get("frame_step", meta.get("frame_step", 1)) or 1))
    return meta, runs


def evaluate_runs(split: SplitData, runs: Dict[str, RecordingRun], eval_cfg: Dict,
                  system_cfg: Dict) -> Tuple[List[RecordingEvaluation], Dict, Dict]:
    missing = [r.recording_id for r in split.recordings if r.recording_id not in runs]
    if missing:
        raise ValueError(f"run is missing recordings that are in the manifest: {missing}")
    extra = sorted(set(runs) - {r.recording_id for r in split.recordings})
    if extra:
        raise ValueError(f"run contains recordings that are not in the {split.name} manifest: {extra}")
    face_loss_cfg = system_cfg.get("face_loss") or {}
    evals = [evaluate_recording(r, split.labels.get(r.recording_id, []), runs[r.recording_id],
                                eval_cfg, face_loss_cfg) for r in split.recordings]
    return evals, aggregate(evals), breakdowns(evals, eval_cfg)


def write_event_tables(run_dir, evals: Sequence[RecordingEvaluation]) -> None:
    """Per-event (matches.csv) and per-alarm (alarms.csv) outcome tables for auditing."""
    run_dir = Path(run_dir)
    with open(run_dir / "matches.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "subject_id", "event_id", "event_type", "start_s", "end_s",
                    "outcome", "alarm_onset_s", "latency_ms", "face_available_ratio"])
        for e in evals:
            if not e.included:
                for p in e.positives:
                    w.writerow([p.recording_id, e.recording.subject_id, p.event_id, p.event_type,
                                p.start_s, p.end_s, f"not_evaluated ({e.status})", "", "", ""])
                continue
            for d in e.match.detections:
                w.writerow([d.event.recording_id, e.recording.subject_id, d.event.event_id, d.event.event_type,
                            d.event.start_s, d.event.end_s, "detected", round(d.episode.onset_s, 4),
                            round(d.latency_ms, 1), round(e.event_face_availability[d.event.event_id], 3)])
            for p in e.match.missed:
                w.writerow([p.recording_id, e.recording.subject_id, p.event_id, p.event_type, p.start_s,
                            p.end_s, "missed", "", "", round(e.event_face_availability[p.event_id], 3)])
    with open(run_dir / "alarms.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["recording_id", "onset_s", "offset_s", "classification", "context"])
        for e in evals:
            if not e.included:
                continue
            m = e.match
            rows = [(d.episode, "detection", d.event.event_id) for d in m.detections]
            rows += [(ep, "false_alarm", ctx) for ep, ctx in zip(m.false_alarms, e.fa_contexts)]
            rows += [(ep, "duplicate", "") for ep in m.duplicates]
            rows += [(ep, "ignored", "") for ep in m.ignored]
            rows += [(ep, "during_calibration", "") for ep in m.before_window]
            for ep, cls, ctx in sorted(rows, key=lambda r: r[0].onset_s):
                w.writerow([e.recording.recording_id, round(ep.onset_s, 4),
                            "" if ep.offset_s is None else round(ep.offset_s, 4), cls, ctx])


def write_outputs(out_dir, *, context: Dict, split: SplitData, summary: Optional[Dict],
                  bd: Optional[Dict], system_cfg: Dict, eval_cfg: Dict, frozen: Optional[Dict]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = dataset_summary(split.recordings, split.labels)
    rp.write_metrics_csv(out_dir / "metrics.csv", summary)
    rp.write_breakdown_csv(out_dir / "breakdown.csv", bd)
    rp.write_report(out_dir / "report.md", rp.render_report(
        context=context, dataset=ds, summary=summary, breakdowns=bd, system_cfg=system_cfg,
        eval_cfg=eval_cfg, frozen=frozen))
    if summary is not None:
        payload = {"context": context, "dataset": ds, "summary": summary, "breakdowns": bd}
        (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def publish(run_dir, results_dir) -> None:
    """Copy an official final result from a run directory to evaluation/results/."""
    for name in ("metrics.csv", "breakdown.csv", "report.md", "summary.json"):
        src = Path(run_dir) / name
        if src.is_file():
            shutil.copy2(src, Path(results_dir) / name)


def write_placeholder(results_dir, final: SplitData, system_cfg: Dict, eval_cfg: Dict,
                      frozen: Optional[Dict]) -> None:
    """Results files stating explicitly that nothing has been measured yet."""
    context = {"generated_at": utc_now(), "split": "final", "run_id": None, "official": False}
    write_outputs(results_dir, context=context, split=final, summary=None, bd=None,
                  system_cfg=system_cfg, eval_cfg=eval_cfg, frozen=frozen)
    stale = Path(results_dir) / "summary.json"
    if stale.is_file():
        stale.unlink()


def official_status(meta: Dict, split: SplitData, frozen_problems: Sequence[str]) -> Tuple[bool, str]:
    """A result is official only for a complete final-split run with valid frozen thresholds."""
    if meta.get("split") != "final":
        return False, "this is a tuning-split run (tuning results are never final results)"
    if frozen_problems:
        return False, "frozen-threshold check failed: " + "; ".join(frozen_problems)
    if meta.get("subset"):
        return False, "only a subset of the final recordings was processed (--only)"
    if int(meta.get("frame_step", 1) or 1) != 1:
        return False, (f"frames were subsampled (--frame-step {meta.get('frame_step')}); this is a "
                       "frame-rate robustness run, not the final result")
    if set(meta["recordings"]) != {r.recording_id for r in split.recordings}:
        return False, "the run does not cover exactly the recordings in the final manifest"
    if meta.get("detection_fingerprint") and meta.get("frozen_detection_fingerprint") and \
            meta["detection_fingerprint"] != meta["frozen_detection_fingerprint"]:
        return False, "run was produced with a configuration different from the frozen one"
    return True, ""
