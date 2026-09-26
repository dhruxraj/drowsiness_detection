"""
Drowsiness score: combines all indicators into one number.

    Total = ACUTE + min(CUMULATIVE, cumulative_cap)

ACUTE (happening now)
    eye     = W_eye  * clamp((closure - blink_max) / (closure_thr - blink_max))
    head    = W_head * clamp((head_down_time - nod_min) / (head_drop_thr - nod_min))
    yawning = W_yawn_now  if a yawn is in progress
CUMULATIVE (recent fatigue history)
    perclos     = W_p * clamp((PERCLOS - low) / (high - low))
    yawns       = W_y * min(yawns_in_window, max)
    nods        = W_n * min(nods_in_window, max)
    long blinks = W_l * min(long_blinks_in_window, max)
    low blink rate = W_b if blink_rate < low_blink_rate

Design consequences (with the default config):
  * A normal blink (< blink_max) adds exactly 0 -> can never trigger the alarm.
  * Eyes closed for closure_duration_threshold -> eye = 100 -> alarm on its own.
  * Fatigue history is capped at 60 < threshold, so it can never trigger the alarm alone,
    but it makes the alarm fire EARLIER (e.g. after 3 yawns a ~1.2 s closure is enough
    instead of 1.8 s).
"""
from __future__ import annotations

from dataclasses import dataclass

from .temporal_analyzer import TemporalState


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


@dataclass
class ScoreBreakdown:
    eye: float = 0.0
    head: float = 0.0
    yawn_now: float = 0.0
    perclos: float = 0.0
    yawns: float = 0.0
    nods: float = 0.0
    long_blinks: float = 0.0
    low_blink_rate: float = 0.0
    acute: float = 0.0
    cumulative: float = 0.0
    total: float = 0.0


class DrowsinessScorer:
    def __init__(self, cfg):
        self.s, self.e, self.h = cfg.scoring, cfg.eyes, cfg.head

    def compute(self, st: TemporalState) -> ScoreBreakdown:
        s, e, h = self.s, self.e, self.h
        b = ScoreBreakdown()

        if st.eyes_closed:
            span = e.closure_duration_threshold - e.blink_max_duration
            b.eye = s.eye_closure_weight * _clamp01((st.closure_duration - e.blink_max_duration) / span)
        if st.head_down:
            span = h.head_drop_duration_threshold - h.nod_min_duration
            b.head = s.head_drop_weight * _clamp01((st.head_down_duration - h.nod_min_duration) / span)
        if st.yawning:
            b.yawn_now = s.yawn_in_progress_weight

        b.perclos = s.perclos_weight * _clamp01(
            (st.perclos - s.perclos_low) / (s.perclos_high - s.perclos_low))
        b.yawns = s.yawn_weight * min(st.yawns_in_window, s.yawn_max_count)
        b.nods = s.nod_weight * min(st.nods_in_window, s.nod_max_count)
        b.long_blinks = s.long_blink_weight * min(st.long_blinks_in_window, s.long_blink_max_count)
        if st.blink_rate is not None and st.blink_rate < e.low_blink_rate:
            b.low_blink_rate = s.low_blink_rate_weight

        b.acute = b.eye + b.head + b.yawn_now
        raw_cum = b.perclos + b.yawns + b.nods + b.long_blinks + b.low_blink_rate
        b.cumulative = min(raw_cum, s.cumulative_cap)
        b.total = b.acute + b.cumulative
        return b
