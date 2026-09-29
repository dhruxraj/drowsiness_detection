"""Writers for ``metrics.csv``, ``breakdown.csv`` and ``report.md``.

When no real labelled recordings have been evaluated, every result cell is written as
``TBD — requires real recording`` (CSV) / ``Not yet measured — real labelled recordings
required`` (Markdown). Nothing is estimated or filled in.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import thresholds as th

TBD_CSV = "TBD — requires real recording"
NOT_MEASURED = "Not yet measured — real labelled recordings required"

CSV_COLUMNS = ("scope", "group", "metric", "value", "unit", "sample_size", "status")

# (metric key in summary / nested path, unit, sample-size key, description)
OVERALL_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("false_alarms_per_hour", "1/h", "non_event_hours"),
    ("false_alarms", "count", "non_event_hours"),
    ("false_alarms_per_evaluated_hour", "1/h", "evaluated_hours"),
    ("missed_events", "count", "positive_events"),
    ("detection_rate_pct", "%", "positive_events"),
    ("latency_ms.mean", "ms", "latency_ms.n"),
    ("latency_ms.median", "ms", "latency_ms.n"),
    ("latency_ms.min", "ms", "latency_ms.n"),
    ("latency_ms.max", "ms", "latency_ms.n"),
    ("latency_ms.p90", "ms", "latency_ms.n"),
    ("measurement_unavailable_rate_pct", "% of frames", "frames_evaluated"),
    ("face_missing_rate_pct", "% of frames", "frames_evaluated"),
    ("invalid_measurement_rate_pct", "% of frames", "frames_evaluated"),
    ("unexpected_unavailable_rate_pct", "% of frames", "frames_evaluated"),
    ("measurement_unavailable_time_pct", "% of time", "frames_evaluated"),
    ("recordings_failed", "count", "recordings_total"),
    ("recordings_no_face", "count", "recordings_total"),
    ("processing_fps_mean", "frames/s", "frames_processed"),
    ("processing_fps_median", "frames/s", "frames_processed"),
    ("processing_fps_p5", "frames/s", "frames_processed"),
    ("processing_fps_min_recording", "frames/s", "recordings_evaluated"),
    ("end_to_end_fps", "frames/s", "frames_processed"),
    ("source_fps_median", "frames/s", "recordings_evaluated"),
)
BREAKDOWN_METRICS = (
    ("positive_events", "count", "positive_events"),
    ("missed_events", "count", "positive_events"),
    ("false_alarms_per_hour", "1/h", "non_event_hours"),
    ("latency_ms.median", "ms", "latency_ms.n"),
    ("measurement_unavailable_rate_pct", "% of frames", "frames_evaluated"),
    ("processing_fps_mean", "frames/s", "frames_processed"),
)


def _get(d: Dict, dotted: str):
    if dotted == "non_event_hours":
        return round(d["non_event_s"] / 3600.0, 4)
    if dotted == "evaluated_hours":
        return round(d["evaluated_s"] / 3600.0, 4)
    node: Any = d
    for p in dotted.split("."):
        node = node.get(p) if isinstance(node, dict) else None
    return node


def _csv_rows(scope: str, group: str, summary: Optional[Dict], metrics, insufficient=False) -> List[List]:
    rows = []
    for key, unit, n_key in metrics:
        name = key.replace(".", "_")
        if summary is None or not summary.get("measured"):
            rows.append([scope, group, name, "", unit, "", TBD_CSV])
            continue
        value = _get(summary, key)
        n = _get(summary, n_key)
        if value is None:
            status = "undefined"
        elif insufficient:
            status = "insufficient_data"
        else:
            status = "measured"
        if isinstance(value, float):
            value = round(value, 4)
        rows.append([scope, group, name, "" if value is None else value, unit,
                     "" if n is None else n, status])
    return rows


def write_metrics_csv(path, summary: Optional[Dict]) -> None:
    _write_csv(path, _csv_rows("overall", "all", summary, OVERALL_METRICS))


def write_breakdown_csv(path, breakdowns: Optional[Dict]) -> None:
    rows: List[List] = []
    for attr, groups in (breakdowns or {}).items():
        for value, s in groups.items():
            insufficient = s.get("insufficient_positive_events") or s.get("insufficient_non_event_time")
            rows.extend(_csv_rows(attr, value, s, BREAKDOWN_METRICS, insufficient))
    if not rows:
        rows = [["(all)", "(none)", m.replace(".", "_"), "", u, "", TBD_CSV]
                for m, u, _ in BREAKDOWN_METRICS]
    _write_csv(path, rows)


def _write_csv(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)


# ---------------------------------------------------------------------------- markdown

def _fmt(value, unit: str = "", digits: int = 2, measured: bool = True, undefined: str = "undefined") -> str:
    if not measured:
        return NOT_MEASURED
    if value is None:
        return undefined
    if isinstance(value, float):
        text = f"{value:.{digits}f}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _latency_text(s: Dict, measured: bool) -> str:
    if not measured:
        return NOT_MEASURED
    lat = s["latency_ms"]
    if not lat["n"]:
        return "undefined (no detected events)"
    return (f"mean {lat['mean']:.0f} ms, median {lat['median']:.0f} ms, "
            f"min {lat['min']:.0f} ms, max {lat['max']:.0f} ms (n={lat['n']})")


def static_limitations() -> List[str]:
    """Limitations that follow from the protocol/framework design and therefore always apply."""
    return [
        "Drowsiness behaviours are **acted** following the recording script (instructed eye "
        "closures, yawns, head drops). Posed events can differ from genuine fatigue, so results "
        "describe detection of the scripted behaviours, not validated drowsiness detection.",
        "Recordings are made at a desk or in a parked vehicle (the project forbids testing while "
        "driving), so vehicle vibration, changing daylight and real driving attention demands are "
        "not represented.",
        "Ground truth is **manually annotated**; event boundaries are limited by frame rate and "
        "annotator judgement. The framework does not compute inter-annotator agreement.",
        "`alarm_expected` encodes the system's intended behaviour (alarm for prolonged eye closure "
        "or sustained head drop). Yawning alone is by design not expected to raise the alarm, so "
        "the evaluation does not judge whether yawning-only fatigue *should* be alarmed.",
        "Alarm latency is measured to the software alarm decision (`alarm_on`), not to physical "
        "buzzer/speaker output; audio, GPIO and serial delays are excluded (see issue #3).",
        "Processing FPS is measured offline on video files without the dashboard or live camera "
        "capture, on the machine that ran the evaluation. It must be re-measured on target "
        "hardware (e.g. Raspberry Pi) before being quoted for that hardware.",
        "The evaluation adapter (`evaluation/core/pipeline.py`) mirrors the per-frame loop of "
        "`main.py` using the same production modules. A drift test guards this, but a change to "
        "`main.py` must be mirrored manually.",
    ]


def data_limitations(dataset: Dict, summary: Optional[Dict], breakdowns: Optional[Dict],
                     eval_cfg: Dict) -> List[str]:
    """Limitations derived from the actual dataset/results (only those that apply)."""
    out: List[str] = []
    if not dataset["recordings"]:
        return ["No real labelled recordings have been evaluated yet; no performance claims can "
                "be made from this report."]
    if dataset["subjects"] < 10:
        out.append(f"Limited number of subjects ({dataset['subjects']}).")
    voc = eval_cfg["vocabulary"]
    for attr in ("lighting", "glasses", "head_position", "camera_distance"):
        missing = [v for v in voc[attr] if v not in dataset["conditions"].get(attr, {})]
        if missing:
            out.append(f"Conditions not covered for `{attr}`: {', '.join(missing)}.")
    if len(dataset["cameras"]) == 1:
        out.append(f"Single camera model ({dataset['cameras'][0]}).")
    hours = dataset["total_duration_s"] / 3600.0
    if hours < 1.0:
        out.append(f"Limited total recording duration ({hours * 60:.1f} min); false-alarm rates per "
                   "hour have wide uncertainty.")
    for etype in ("blink", "eye_closure", "yawn", "head_movement", "head_drop", "face_loss"):
        if etype not in dataset["event_counts"]:
            out.append(f"No labelled `{etype}` events.")
    if breakdowns:
        thin = [f"{a}={v}" for a, g in breakdowns.items() for v, s in g.items()
                if s.get("insufficient_positive_events")]
        if thin:
            out.append("Too few alarm-expected events for a reliable breakdown in: " + ", ".join(thin) + ".")
    if summary and summary.get("recordings_failed"):
        out.append(f"{summary['recordings_failed']} recording(s) failed to process and are excluded "
                   "from the alarm metrics (listed in the results).")
    return out


def render_report(*, context: Dict, dataset: Dict, summary: Optional[Dict], breakdowns: Optional[Dict],
                  system_cfg: Dict, eval_cfg: Dict, frozen: Optional[Dict]) -> str:
    measured = bool(summary and summary.get("measured"))
    m = eval_cfg["matching"]
    L: List[str] = ["# Multi-condition evaluation report", ""]
    if not measured:
        L += [f"> **Status: {NOT_MEASURED}.** This file documents the evaluation set-up. "
              "It contains no experimental results.", ""]
    elif context.get("official"):
        L += ["> **Status: final evaluation on the held-out FINAL split with frozen thresholds.**", ""]
    else:
        L += [f"> **Status: NOT an official final result** - {context.get('not_official_reason', '')}", ""]
    L += [_table(["Item", "Value"], [
        ["Generated (UTC)", context.get("generated_at", "")],
        ["Split", context.get("split", "final")],
        ["Run ID", context.get("run_id") or "—"],
        ["Git commit", (context.get("git") or {}).get("commit") or "—"],
        ["Landmark engine", context.get("landmark_engine") or "—"],
        ["Frames processed", "every frame" if int(context.get("frame_step") or 1) == 1
         else f"1 of every {context.get('frame_step')} frames (frame-rate robustness run)"],
        ["Detection config fingerprint", th.fingerprint(th.detection_snapshot(system_cfg))[:16]],
    ]), ""]

    # ---- dataset
    L += ["## 1. Dataset", ""]
    if not dataset["recordings"]:
        L += [f"No labelled recordings are present in this split yet. {NOT_MEASURED}.", ""]
    L += [_table(["Quantity", "Value"], [
        ["Subjects", dataset["subjects"]],
        ["Recordings", dataset["recordings"]],
        ["Total duration", f"{dataset['total_duration_s'] / 60.0:.1f} min"],
        ["Camera models", ", ".join(dataset["cameras"]) or "—"],
        ["Resolutions", ", ".join(dataset["resolutions"]) or "—"],
        ["Source FPS", ", ".join(f"{x:g}" for x in dataset["source_fps"]) or "—"],
    ]), ""]
    for attr in ("lighting", "glasses", "head_position", "camera_distance"):
        rows = [[k, v["recordings"], v["subjects"], f"{v['duration_s'] / 60.0:.1f}"]
                for k, v in dataset["conditions"].get(attr, {}).items()] or [["—", 0, 0, "0.0"]]
        L += [f"**{attr}**", "", _table([attr, "recordings", "subjects", "minutes"], rows), ""]
    rows = [[k, v["count"], f"{v['duration_s']:.1f}", v["yes"], v["no"], v["ignore"]]
            for k, v in dataset["event_counts"].items()] or [["—", 0, "0.0", 0, 0, 0]]
    L += ["**Labelled events**", "",
          _table(["event_type", "count", "total s", "alarm_expected=yes", "no", "ignore"], rows), ""]

    # ---- protocol
    L += ["## 2. Protocol", "",
          "- Tuning and final recordings are kept in separate manifests; the evaluator rejects any "
          "shared recording ID, subject ID, file path or file hash.",
          f"- Event matching: an alarm onset detects an alarm-expected event if it lies in "
          f"[start − {m['pre_tolerance_s']} s, end + {m['post_tolerance_s']} s]; one onset per event.",
          "- The labelled calibration period at the start of each recording is excluded from all "
          "alarm and face-loss metrics.",
          "- Full definitions: `evaluation/README.md`.", ""]
    tuning_ids = (frozen or {}).get("tuning_recordings") or []
    tuning_subjects = (frozen or {}).get("tuning_subjects") or []
    L += ["**Recordings used**", "", _table(["Split", "Subjects", "Recordings"], [
        ["tuning (threshold selection, from the frozen record)",
         ", ".join(tuning_subjects) or "—", ", ".join(tuning_ids) or "—"],
        [f"{context.get('split', 'final')} (this report)",
         ", ".join(dataset.get("subject_ids") or []) or "—",
         ", ".join(dataset.get("recording_ids") or []) or "—"],
    ]), ""]

    # ---- thresholds
    L += ["## 3. Thresholds", ""]
    if frozen and frozen.get("status") == th.FROZEN:
        L += [f"Frozen at {frozen.get('frozen_at')} (commit {frozen.get('git_commit') or '—'}), "
              f"selection: {frozen.get('selection_method')}, tuning recordings: "
              f"{len(frozen.get('tuning_recordings') or [])}, tuning subjects: "
              f"{len(frozen.get('tuning_subjects') or [])}. Detection fingerprint "
              f"`{frozen.get('detection_fingerprint', '')[:16]}`.", ""]
    else:
        L += ["**Thresholds are not frozen yet.** The values below are the current `config.yaml` "
              "defaults; they have not been tuned on real labelled recordings.", ""]
    L += [_table(["Parameter", "Value", "Why it exists"],
                 [[f"`{r['key']}`", r["value"], r["rationale"]] for r in th.key_thresholds(system_cfg)]), ""]

    # ---- results
    s = summary or {}
    L += ["## 4. Results", ""]
    fa_text = _fmt(s.get("false_alarms_per_hour"), "/h", measured=measured,
                   undefined="undefined (no non-event time)")
    if measured and s.get("false_alarms_per_hour") is not None:
        fa_text += f" ({s['false_alarms']} false alarms in {s['non_event_s'] / 3600.0:.2f} h non-event time)"
    missed_text = NOT_MEASURED if not measured else (
        f"{s['missed_events']} of {s['positive_events']} alarm-expected events "
        f"(detection rate {_fmt(s['detection_rate_pct'], '%', 1)})")
    loss_text = NOT_MEASURED if not measured else (
        f"{_fmt(s['measurement_unavailable_rate_pct'], '%')} of frames unavailable "
        f"(face missing {_fmt(s['face_missing_rate_pct'], '%')}, invalid {_fmt(s['invalid_measurement_rate_pct'], '%')}; "
        f"outside labelled face-loss periods {_fmt(s['unexpected_unavailable_rate_pct'], '%')}); "
        f"complete recording failures: {s['recordings_failed']}")
    fps_text = NOT_MEASURED if not measured else (
        f"mean {_fmt(s['processing_fps_mean'], 'fps', 1)}, median {_fmt(s['processing_fps_median'], 'fps', 1)}, "
        f"5th percentile {_fmt(s['processing_fps_p5'], 'fps', 1)} (source video {_fmt(s['source_fps_median'], 'fps', 1)})")
    L += [_table(["Metric", "Result"], [
        ["False alarms/hour", fa_text],
        ["Missed events", missed_text],
        ["Alarm latency", _latency_text(s, measured)],
        ["Face/measurement loss rate", loss_text],
        ["Processing FPS", fps_text],
    ]), ""]

    if measured:
        L += ["### Supporting results", "",
              _table(["Quantity", "Value"], [
                  ["Recordings evaluated / failed / face never found",
                   f"{s['recordings_evaluated']} / {s['recordings_failed']} / {s['recordings_no_face']}"],
                  ["Evaluated time", f"{s['evaluated_s'] / 3600.0:.2f} h"],
                  ["Duplicate alarms (same event)", s["duplicate_alarms"]],
                  ["Alarms in ignore regions", s["ignored_alarms"]],
                  ["Alarms during calibration", s["alarms_during_calibration"]],
                  ["False alarms per evaluated hour (all time)", _fmt(s["false_alarms_per_evaluated_hour"], "/h")],
                  ["Missed while face/measurement mostly unavailable", s["missed_while_face_unavailable"]],
                  ["Alarm recovery time: alarm off after event end (median)", _fmt(s["release_delay_ms"]["median"], "ms", 0)],
                  ["End-to-end throughput incl. decoding", _fmt(s["end_to_end_fps"], "fps", 1)],
                  ["Lowest per-recording processing FPS", _fmt(s["processing_fps_min_recording"], "fps", 1)],
                  ["Face-loss episodes (short / temporary / extended)",
                   " / ".join(str(s["face_loss_episodes"][k]) for k in ("short_dropout", "temporary", "extended"))],
                  ["Positive events not evaluated (failed recordings)", s["positive_events_not_evaluated"]],
              ]), ""]
        rows = [[k, v["events"], v["detected"], v["missed"], _fmt(v["latency_median_ms"], "ms", 0)]
                for k, v in s["by_event_type"].items()] or [["—", 0, 0, 0, "—"]]
        L += ["### Detection by event type", "", _table(["event_type", "events", "detected", "missed", "median latency"], rows), ""]
        rows = [[k, v] for k, v in s["false_alarms_by_context"].items()] or [["—", 0]]
        L += ["### False alarms by labelled context", "", _table(["context at alarm onset", "false alarms"], rows), ""]
        rows = [[k, v["detected"], v["events"], _fmt(v["recall_pct"], "%", 1)] for k, v in s["indicator_recall"].items()]
        if rows:
            L += ["### Supplementary: analyzer indicator events", "",
                  "Whether the temporal analyzer logged the matching indicator (e.g. `YAWN STARTED`) "
                  "during each labelled event. Not an alarm metric.", "",
                  _table(["event_type", "logged", "labelled", "recall"], rows), ""]
        if s["failed_recordings"]:
            L += ["### Failed recordings", "", _table(["recording", "status"], list(s["failed_recordings"].items())), ""]

    # ---- breakdowns
    L += ["## 5. Breakdown by condition", ""]
    if not measured or not breakdowns:
        L += [NOT_MEASURED + ".", ""]
    else:
        for attr, groups in breakdowns.items():
            rows = []
            for value, g in groups.items():
                flag = " ⚠ insufficient data" if (g["insufficient_positive_events"] or g["insufficient_non_event_time"]) else ""
                rows.append([f"{value}{flag}", g["recordings_evaluated"], g["positive_events"],
                             "—" if g["missed_events"] is None else g["missed_events"],
                             _fmt(g["false_alarms_per_hour"], "/h"),
                             _fmt(g["latency_ms"]["median"], "ms", 0),
                             _fmt(g["measurement_unavailable_rate_pct"], "%"),
                             _fmt(g["processing_fps_mean"], "fps", 1)])
            L += [f"**{attr}**", "", _table([attr, "recordings", "alarm-expected events", "missed",
                                              "false alarms/h", "median latency", "measurement loss", "processing FPS"], rows), ""]
        L += [f"Groups marked ⚠ have fewer than {eval_cfg['breakdowns']['min_positive_events']} alarm-expected "
              f"events or less than {eval_cfg['breakdowns']['min_non_event_minutes']} min of non-event time.", ""]

    # ---- environment
    env = context.get("environment") or {}
    L += ["## 6. Evaluation environment", ""]
    if env:
        L += [_table(["Item", "Value"], [["Python", env.get("python")], ["Platform", env.get("platform")],
                                         ["CPU count", env.get("cpu_count")]] +
                     [[k, v or "not installed"] for k, v in (env.get("packages") or {}).items()]), ""]
    else:
        L += ["No evaluation run yet. The machine, Python and package versions are recorded "
              "automatically with every run (`run.json`).", ""]

    # ---- limitations
    L += ["## 7. Limitations", "", "**Always applicable (protocol/framework design):**", ""]
    L += [f"- {x}" for x in static_limitations()] + [""]
    L += ["**Specific to this dataset/run:**", ""]
    L += [f"- {x}" for x in data_limitations(dataset, summary, breakdowns, eval_cfg)] + [""]
    if measured and s.get("warnings"):
        L += ["**Warnings raised during the run:**", ""] + [f"- {w}" for w in s["warnings"]] + [""]

    L += ["## 8. Reproduce", "", "See `evaluation/README.md` → *Reproducing the evaluation*.", ""]
    return "\n".join(L)


def write_report(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
