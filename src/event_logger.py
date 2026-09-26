"""
Event logging.

Writes two files per session in logs/:
  session_YYYYmmdd_HHMMSS.log  human-readable, e.g.
        12:35:21 | EAR: 0.31 | MAR: 0.42 | SCORE:   0.0 | STATUS: ALERT
        12:35:30 | ALARM ACTIVATED
  session_YYYYmmdd_HHMMSS.csv  one row per logged sample (for Excel / plotting)
A line is written on every state change, on every event, and every `interval` seconds.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime


class EventLogger:
    def __init__(self, cfg, session_start: datetime | None = None):
        self.interval = float(cfg.interval)
        self.console = bool(cfg.console)
        os.makedirs(cfg.directory, exist_ok=True)
        stamp = (session_start or datetime.now()).strftime("%Y%m%d_%H%M%S")
        base = os.path.join(cfg.directory, f"session_{stamp}")
        self.log_path = base + ".log"
        self._txt = open(self.log_path, "w", encoding="utf-8")
        self._csv_file = None
        self._csv = None
        if cfg.csv:
            self.csv_path = base + ".csv"
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["timestamp", "ear", "mar", "pitch", "yaw", "perclos",
                                "blink_rate", "score", "acute", "cumulative", "state", "alarm"])
        self._last_state = None
        self._last_t = -1e9
        self.alarm_activations = 0

    @staticmethod
    def _fmt(v, spec=".2f") -> str:
        return "--" if v is None else format(v, spec)

    def _write(self, line: str) -> None:
        self._txt.write(line + "\n")
        self._txt.flush()
        if self.console:
            print(line, flush=True)

    def log_sample(self, t: float, wall: datetime, st, score, state: str, alarm_on: bool) -> None:
        """Log a measurement line if the state changed or the interval elapsed."""
        if state == self._last_state and t - self._last_t < self.interval:
            return
        self._last_state, self._last_t = state, t
        ear = st.ear if st is not None else None
        mar = st.mar if st is not None else None
        sc = score.total if score is not None else 0.0
        self._write(f"{wall:%H:%M:%S} | EAR: {self._fmt(ear)} | MAR: {self._fmt(mar)} | "
                    f"SCORE: {sc:5.1f} | STATUS: {state}")
        if self._csv is not None:
            self._csv.writerow([
                wall.isoformat(timespec="milliseconds"),
                self._fmt(ear, ".4f"), self._fmt(mar, ".4f"),
                self._fmt(st.pitch_rel if st else None, ".1f"),
                self._fmt(st.yaw_rel if st else None, ".1f"),
                self._fmt(st.perclos if st else None, ".3f"),
                self._fmt(st.blink_rate if st else None, ".1f"),
                f"{sc:.1f}",
                f"{score.acute:.1f}" if score else "0.0",
                f"{score.cumulative:.1f}" if score else "0.0",
                state, int(alarm_on)])
            self._csv_file.flush()

    def log_event(self, wall: datetime, message: str) -> None:
        if message == "ALARM ACTIVATED":
            self.alarm_activations += 1
        self._write(f"{wall:%H:%M:%S} | {message}")

    def close(self, summary: str | None = None) -> None:
        if summary:
            self._write(summary)
        self._txt.close()
        if self._csv_file:
            self._csv_file.close()
