"""Freezing detection thresholds before the final evaluation.

``freeze_thresholds.py`` writes ``evaluation/config/frozen_thresholds.yaml`` containing a
snapshot of every detection-relevant section of ``config.yaml`` plus the evaluation
protocol (matching tolerances, ground-truth definitions) and SHA-256 fingerprints of
both. ``evaluate.py --split final`` refuses to run unless the current configuration
still matches the frozen fingerprints, so thresholds cannot silently change after tuning.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

# Sections of config.yaml that influence detection results.
DETECTION_SECTIONS = ("preprocessing", "face_mesh", "eyes", "mouth", "head", "face_loss",
                      "calibration", "scoring")
DETECTION_ALARM_KEYS = ("min_duration", "release_time")  # alarm timing only, not hardware
PROTOCOL_SECTIONS = ("matching", "ground_truth")

FROZEN = "frozen"
NOT_FROZEN = "not_frozen"

# Why each key threshold exists (summarised from config.yaml comments and README section 10).
THRESHOLD_RATIONALE = {
    "eyes.ear_threshold": "Fallback closed-eye EAR limit when calibration is off or fails.",
    "calibration.ear_ratio": "Personal closed-eye threshold = ratio x the driver's open-eye EAR "
                             "(eye shape differs between people).",
    "calibration.ear_min": "Lower clamp of the personal EAR threshold.",
    "calibration.ear_max": "Upper clamp of the personal EAR threshold.",
    "eyes.ear_hysteresis": "Eyes count as re-opened only above threshold + hysteresis (prevents flicker).",
    "eyes.blink_max_duration": "Closures shorter than this are normal blinks and add 0 to the eye score.",
    "eyes.long_blink_min_duration": "Closures at least this long count as long blinks (fatigue history).",
    "eyes.closure_duration_threshold": "Continuous closure that on its own raises the alarm (micro-sleep).",
    "eyes.max_yaw_for_eye_analysis": "Beyond this head turn EAR is unreliable and eye evidence is ignored.",
    "mouth.mar_threshold": "MAR above this = mouth wide open (yawn candidate).",
    "calibration.mar_margin": "Personal yawn threshold = max(mar_threshold, resting MAR + margin).",
    "mouth.yawn_min_duration": "Mouth must stay open this long to count as a yawn (talking is shorter).",
    "head.pitch_down_threshold": "Head pitched down more than this vs. calibrated neutral = head drop.",
    "head.nod_min_duration": "Head dips shorter than this are ignored (bumps, glances).",
    "head.head_drop_duration_threshold": "Sustained head drop that on its own raises the alarm.",
    "head.yaw_distraction_threshold": "Beyond this yaw the state is LOOKING AWAY (never drowsiness).",
    "face_loss.grace_period": "Face dropouts shorter than this are ignored.",
    "face_loss.warn_after": "Face missing this long -> DRIVER NOT VISIBLE warning (no drowsiness alarm).",
    "scoring.threshold": "Total drowsiness score at/above which the alarm switches on.",
    "scoring.release_threshold": "Acute score must fall below this before the alarm can stop.",
    "scoring.cumulative_cap": "Cap on fatigue-history score so history alone never reaches the threshold.",
    "alarm.min_duration": "Once on, the alarm sounds at least this long.",
    "alarm.release_time": "Eyes open / head up this long before the alarm stops.",
}


def _get(data: Dict[str, Any], dotted: str):
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def detection_snapshot(system_cfg: Dict[str, Any]) -> Dict[str, Any]:
    snap = {k: copy.deepcopy(system_cfg.get(k)) for k in DETECTION_SECTIONS}
    alarm = system_cfg.get("alarm") or {}
    snap["alarm"] = {k: alarm.get(k) for k in DETECTION_ALARM_KEYS}
    return snap


def protocol_snapshot(eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: copy.deepcopy(eval_cfg.get(k)) for k in PROTOCOL_SECTIONS}


def fingerprint(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def key_thresholds(system_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows (key, value, rationale) for the documented key thresholds."""
    return [{"key": k, "value": _get(system_cfg, k), "rationale": r}
            for k, r in THRESHOLD_RATIONALE.items()]


def build_frozen_record(system_cfg: Dict[str, Any], eval_cfg: Dict[str, Any], *,
                        tuning_recording_ids: Sequence[str], tuning_subject_ids: Sequence[str],
                        tuning_manifest_sha256: Optional[str], selection_method: str,
                        tuning_run: Optional[str], git_commit: Optional[str], frozen_at: str,
                        notes: str, final_results_existed: bool) -> Dict[str, Any]:
    det = detection_snapshot(system_cfg)
    proto = protocol_snapshot(eval_cfg)
    return {
        "status": FROZEN,
        "frozen_at": frozen_at,
        "git_commit": git_commit,
        "selection_method": selection_method,
        "tuning_run": tuning_run,
        "tuning_recordings": sorted(tuning_recording_ids),
        "tuning_subjects": sorted(set(tuning_subject_ids)),
        "tuning_manifest_sha256": tuning_manifest_sha256,
        "final_results_existed_at_freeze": final_results_existed,
        "notes": notes,
        "detection_fingerprint": fingerprint(det),
        "protocol_fingerprint": fingerprint(proto),
        "detection_config": det,
        "protocol": proto,
    }


def verify_frozen(frozen: Optional[Dict[str, Any]], system_cfg: Dict[str, Any],
                  eval_cfg: Dict[str, Any], final_recording_ids: Sequence[str],
                  final_subject_ids: Sequence[str]) -> List[str]:
    """Problems that make a final evaluation invalid (empty list = OK)."""
    if not frozen or frozen.get("status") != FROZEN:
        return ["thresholds are not frozen - tune on the tuning split, then run "
                "evaluation/scripts/freeze_thresholds.py before evaluating the final split"]
    problems = []
    det = detection_snapshot(system_cfg)
    if fingerprint(det) != frozen.get("detection_fingerprint"):
        changed = [k for k in det if det.get(k) != (frozen.get("detection_config") or {}).get(k)]
        problems.append("config.yaml detection settings differ from the frozen snapshot "
                        f"(changed section(s): {changed or 'unknown'}); thresholds must not change "
                        "after freezing")
    if fingerprint(protocol_snapshot(eval_cfg)) != frozen.get("protocol_fingerprint"):
        problems.append("evaluation protocol (matching tolerances / ground-truth definitions) "
                        "differs from the frozen snapshot")
    leak_rec = sorted({r.lower() for r in frozen.get("tuning_recordings") or []}
                      & {r.lower() for r in final_recording_ids})
    if leak_rec:
        problems.append(f"final recordings were used for tuning: {leak_rec}")
    leak_sub = sorted({s.lower() for s in frozen.get("tuning_subjects") or []}
                      & {s.lower() for s in final_subject_ids})
    if leak_sub:
        problems.append(f"final subjects were used for tuning: {leak_sub}")
    return problems
