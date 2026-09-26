"""
Configuration loader.

Reads config.yaml and exposes it with attribute access, e.g.
    cfg.eyes.ear_threshold
A few sanity checks catch common editing mistakes early, with a clear message.
"""
from __future__ import annotations

import os
from typing import Any

import yaml


class ConfigSection(dict):
    """A dict that also allows attribute access (cfg.section.key)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Missing configuration key: '{key}'") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigSection({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


class ConfigError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(f"Invalid configuration: {msg}")


def validate(cfg: ConfigSection) -> None:
    e, m, h, s = cfg.eyes, cfg.mouth, cfg.head, cfg.scoring
    _check(0.05 < e.ear_threshold < 0.5, "eyes.ear_threshold should be between 0.05 and 0.5")
    _check(0 < e.smoothing_alpha <= 1, "eyes.smoothing_alpha must be in (0, 1]")
    _check(e.blink_max_duration < e.closure_duration_threshold,
           "eyes.blink_max_duration must be shorter than eyes.closure_duration_threshold")
    _check(e.long_blink_min_duration >= e.blink_max_duration,
           "eyes.long_blink_min_duration must be >= eyes.blink_max_duration")
    _check(0.1 < m.mar_threshold < 2.0, "mouth.mar_threshold should be between 0.1 and 2.0")
    _check(h.nod_min_duration < h.head_drop_duration_threshold,
           "head.nod_min_duration must be shorter than head.head_drop_duration_threshold")
    _check(s.release_threshold < s.threshold, "scoring.release_threshold must be < scoring.threshold")
    _check(s.cumulative_cap < s.threshold,
           "scoring.cumulative_cap must be < scoring.threshold (history alone must not trigger the alarm)")
    _check(isinstance(cfg.alarm.backends, list) and cfg.alarm.backends,
           "alarm.backends must be a non-empty list")


def load_config(path: str = "config.yaml") -> ConfigSection:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _wrap(raw)
    validate(cfg)
    return cfg
