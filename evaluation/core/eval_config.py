"""Loader/validator for ``evaluation/config/evaluation_config.yaml``.

The evaluation config holds the *protocol* (paths, label vocabulary, event-matching
tolerances, breakdown settings, tuning search space). Detection thresholds are NOT
defined here; they live in the production ``config.yaml`` and are frozen with
``evaluation/scripts/freeze_thresholds.py``.
"""
from __future__ import annotations

import copy
import numbers
from pathlib import Path
from typing import Any, Dict, List, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG = REPO_ROOT / "evaluation" / "config" / "evaluation_config.yaml"

ALARM_EXPECTED_VALUES = ("yes", "no", "ignore")

DEFAULTS: Dict[str, Any] = {
    "paths": {
        "system_config": "config.yaml",
        "frozen_thresholds": "evaluation/config/frozen_thresholds.yaml",
        "results_dir": "evaluation/results",
        "runs_dir": "evaluation/results/runs",
        "splits": {
            "tuning": {
                "dataset_dir": "evaluation/dataset/tuning",
                "recordings": "evaluation/labels/tuning_recordings.csv",
                "labels": "evaluation/labels/tuning_labels.csv",
            },
            "final": {
                "dataset_dir": "evaluation/dataset/final",
                "recordings": "evaluation/labels/evaluation_recordings.csv",
                "labels": "evaluation/labels/evaluation_labels.csv",
            },
        },
    },
    "vocabulary": {
        "lighting": ["normal_indoor", "dim_indoor", "bright_frontal", "backlit", "side_lit",
                     "night_ir", "daylight_vehicle"],
        "glasses": ["none", "clear", "sunglasses"],
        "head_position": ["frontal", "yaw_offset", "pitch_offset", "varied"],
        "camera_distance": ["near", "medium", "far"],
        "event_types": {
            "normal": ["no"],
            "blink": ["no"],
            "eye_closure": ["yes", "no", "ignore"],
            "yawn": ["no", "ignore"],
            "talking": ["no", "ignore"],
            "head_movement": ["no", "ignore"],
            "head_drop": ["yes", "no", "ignore"],
            "face_loss": ["no", "ignore"],
            "other": ["no", "ignore"],
        },
    },
    "ground_truth": {
        "prolonged_closure_min_s": 3.0,
        "brief_closure_max_s": 1.0,
        "sustained_head_drop_min_s": 3.0,
    },
    "matching": {
        "pre_tolerance_s": 0.5,
        "post_tolerance_s": 1.0,
    },
    "breakdowns": {
        "attributes": ["subject_id", "lighting", "glasses", "head_position", "camera_distance"],
        "min_positive_events": 5,
        "min_non_event_minutes": 10.0,
    },
    "recording_checks": {
        "max_frame_shortfall_ratio": 0.02,
    },
    "runner": {
        "read_timeout_s": 1.0,
        "stall_timeout_s": 15.0,
    },
    "indicator_events": {
        "yawn": ["^YAWN STARTED"],
        "eye_closure": ["^MICROSLEEP", "^LONG BLINK"],
        "head_drop": ["^HEAD DROP"],
    },
    "tuning": {
        "max_false_alarms_per_hour": 1.0,
        "search_space": {},
    },
}

# Parameters that act *before* the per-frame measurements are produced. Changing them
# changes the recorded EAR/MAR/pose values, so they cannot be tuned by replaying traces.
REPLAY_UNSAFE_PREFIXES = ("camera.", "preprocessing.", "face_mesh.", "head.invert_pitch",
                          "head.enabled")


class EvalConfigError(ValueError):
    pass


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "event_types" \
                and key != "search_space" and key != "indicator_events":
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _nonneg_number(value: Any, name: str, errors: List[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or value < 0:
        errors.append(f"{name} must be a non-negative number (got {value!r})")


def _normalize_yaml_booleans(cfg: Dict[str, Any]) -> None:
    """YAML 1.1 parses unquoted yes/no as True/False; map them back to label strings."""
    etypes = cfg.get("vocabulary", {}).get("event_types")
    if isinstance(etypes, dict):
        for name, allowed in list(etypes.items()):
            if isinstance(allowed, list):
                etypes[name] = [("yes" if a else "no") if isinstance(a, bool) else a
                                for a in allowed]


def validate(cfg: Dict[str, Any]) -> None:
    errors: List[str] = []
    m = cfg["matching"]
    _nonneg_number(m.get("pre_tolerance_s"), "matching.pre_tolerance_s", errors)
    _nonneg_number(m.get("post_tolerance_s"), "matching.post_tolerance_s", errors)
    gt = cfg["ground_truth"]
    for key in ("prolonged_closure_min_s", "brief_closure_max_s", "sustained_head_drop_min_s"):
        _nonneg_number(gt.get(key), f"ground_truth.{key}", errors)
    voc = cfg["vocabulary"]
    for key in ("lighting", "glasses", "head_position", "camera_distance"):
        vals = voc.get(key)
        if not isinstance(vals, list) or not vals or not all(isinstance(v, str) for v in vals):
            errors.append(f"vocabulary.{key} must be a non-empty list of strings")
    etypes = voc.get("event_types")
    if not isinstance(etypes, dict) or not etypes:
        errors.append("vocabulary.event_types must be a non-empty mapping")
    else:
        for name, allowed in etypes.items():
            if not isinstance(allowed, list) or not allowed or \
                    any(a not in ALARM_EXPECTED_VALUES for a in allowed):
                errors.append(f"vocabulary.event_types.{name} must list allowed alarm_expected "
                              f"values from {ALARM_EXPECTED_VALUES}")
    b = cfg["breakdowns"]
    _nonneg_number(b.get("min_positive_events"), "breakdowns.min_positive_events", errors)
    _nonneg_number(b.get("min_non_event_minutes"), "breakdowns.min_non_event_minutes", errors)
    r = cfg["runner"]
    for key in ("read_timeout_s", "stall_timeout_s"):
        _nonneg_number(r.get(key), f"runner.{key}", errors)
    _nonneg_number(cfg["tuning"].get("max_false_alarms_per_hour"),
                   "tuning.max_false_alarms_per_hour", errors)
    space = cfg["tuning"].get("search_space") or {}
    if not isinstance(space, dict):
        errors.append("tuning.search_space must be a mapping of dotted config keys to value lists")
    else:
        for key, values in space.items():
            if not isinstance(values, list) or not values:
                errors.append(f"tuning.search_space.{key} must be a non-empty list")
            if any(key == p.rstrip(".") or key.startswith(p) for p in REPLAY_UNSAFE_PREFIXES):
                errors.append(f"tuning.search_space.{key}: this parameter changes the per-frame "
                              "measurements and cannot be tuned by trace replay")
    ind = cfg.get("indicator_events") or {}
    if not isinstance(ind, dict):
        errors.append("indicator_events must be a mapping of event_type to regex list")
    if errors:
        raise EvalConfigError("Invalid evaluation config:\n  " + "\n  ".join(errors))


def from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _deep_merge(DEFAULTS, data or {})
    _normalize_yaml_booleans(cfg)
    validate(cfg)
    return cfg


def load(path: Union[str, Path, None] = None) -> Dict[str, Any]:
    import yaml  # PyYAML is already a project dependency

    path = Path(path) if path else DEFAULT_EVAL_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise EvalConfigError(f"{path}: top level must be a mapping")
    return from_dict(data)


def resolve(path: Union[str, Path]) -> Path:
    """Resolve a repository-relative path from the config to an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)
