"""
Temporal analysis: turns noisy per-frame measurements into drowsiness evidence.

Key ideas (all time-based, so behaviour does not depend on the camera FPS):
  * Eye closure is measured as a DURATION. A closure is classified when the eye re-opens:
        < long_blink_min_duration          -> normal blink
        >= long_blink_min_duration         -> long blink (fatigue sign)
        >= closure_duration_threshold      -> micro-sleep
  * PERCLOS  = fraction of time the eyes were closed over the last `perclos_window` seconds.
  * Blink rate (blinks/min) over `blink_rate_window`; a very LOW rate is a fatigue sign.
  * Yawn  = MAR above threshold continuously for >= yawn_min_duration (talking is shorter).
  * Nod / head drop = pitch below the calibrated neutral by > pitch_down_threshold for
    >= nod_min_duration.
Robustness rules:
  * Hysteresis on every threshold (no flicker around the threshold).
  * Face lost for < grace_period -> everything is simply paused.
  * Face lost for longer -> ongoing episodes are discarded (a detection failure is never
    counted as drowsiness).
  * Head turned sideways (|yaw| > max_yaw_for_eye_analysis) -> EAR is unreliable, so eye
    evidence is ignored instead of being read as "eyes closed".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .metrics import EMA

MAX_DT = 0.25  # s - a longer gap between frames is clipped (e.g. a frozen camera)


@dataclass
class FrameMetrics:
    ear: float
    mar: float
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None


@dataclass
class TemporalState:
    time: float = 0.0
    face_visible: bool = False
    face_lost_duration: float = 0.0
    ear: float | None = None
    mar: float | None = None
    pitch_rel: float = 0.0
    yaw_rel: float = 0.0
    eye_analysis_valid: bool = False
    eyes_closed: bool = False
    closure_duration: float = 0.0
    eyes_open_duration: float = 0.0
    recent_blink: bool = False
    blink_rate: float | None = None       # blinks/min (None until enough data)
    blinks_in_window: int = 0
    long_blinks_in_window: int = 0
    perclos: float = 0.0
    yawning: bool = False
    mouth_open_duration: float = 0.0
    yawns_in_window: int = 0
    head_down: bool = False
    head_down_duration: float = 0.0
    nods_in_window: int = 0
    looking_away: bool = False
    total_blinks: int = 0
    total_yawns: int = 0
    total_nods: int = 0
    events: list[str] = field(default_factory=list)


class TemporalAnalyzer:
    def __init__(self, cfg):
        self.e, self.m, self.h, self.fl = cfg.eyes, cfg.mouth, cfg.head, cfg.face_loss
        # thresholds that calibration may personalise
        self.ear_threshold = float(self.e.ear_threshold)
        self.mar_threshold = float(self.m.mar_threshold)
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0
        self.reset()

    # ------------------------------------------------------------------ setup
    def apply_calibration(self, ear_threshold, mar_threshold, pitch0, yaw0) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.pitch_offset = pitch0
        self.yaw_offset = yaw0
        self.reset()

    def reset(self) -> None:
        self._ear_ema = EMA(float(self.e.smoothing_alpha))
        self._mar_ema = EMA(float(self.m.smoothing_alpha))
        self._last_t: float | None = None
        self._start_t: float | None = None
        self._face_lost = 0.0
        self._eyes_closed = False
        self._closure = 0.0
        self._eyes_open = 0.0
        self._blink_shown_until = -1.0
        self._mouth_open = 0.0
        self._yawning = False
        self._head_down_t = 0.0
        self._head_down = False
        self._eye_samples: deque = deque()     # (t, dt, closed)
        self._eye_observed = 0.0               # total seconds of valid eye data
        self._blinks: deque = deque()
        self._long_blinks: deque = deque()
        self._yawns: deque = deque()
        self._nods: deque = deque()
        self.total_blinks = self.total_yawns = self.total_nods = 0
        self._pitch_rel = 0.0
        self._yaw_rel = 0.0
        self._last_state = TemporalState()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _prune(dq: deque, t: float, window: float) -> None:
        while dq and dq[0] < t - window:
            dq.popleft()

    def _end_eye_episode(self, t: float, events: list[str]) -> None:
        """Classify a finished eye closure."""
        d = self._closure
        if d <= 0:
            return
        if d < self.e.long_blink_min_duration:
            self._blinks.append(t)
            self.total_blinks += 1
            self._blink_shown_until = t + 0.3
        else:
            self._long_blinks.append(t)
            kind = "MICROSLEEP" if d >= self.e.closure_duration_threshold else "LONG BLINK"
            events.append(f"{kind} ({d:.2f}s)")

    def _discard_episodes(self) -> None:
        self._eyes_closed = False
        self._closure = 0.0
        self._eyes_open = 0.0
        self._mouth_open = 0.0
        self._yawning = False
        self._head_down = False
        self._head_down_t = 0.0

    # ------------------------------------------------------------------ main update
    def update(self, m: FrameMetrics | None, t: float) -> TemporalState:
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), MAX_DT)
        self._last_t = t
        if self._start_t is None:
            self._start_t = t
        events: list[str] = []

        # ---------------- face not detected
        if m is None:
            self._face_lost += dt
            if self._face_lost >= self.fl.grace_period and (self._eyes_closed or self._yawning or self._head_down):
                self._discard_episodes()  # never turn a detection failure into drowsiness
            if self._face_lost >= self.fl.warn_after and self._face_lost - dt < self.fl.warn_after:
                events.append("DRIVER NOT VISIBLE")
            s = self._build_state(t, face_visible=self._face_lost < self.fl.grace_period,
                                  ear=None, mar=None, eye_valid=False, events=events)
            return s
        if self._face_lost >= self.fl.grace_period:
            events.append(f"FACE REACQUIRED after {self._face_lost:.1f}s")
        self._face_lost = 0.0

        ear = self._ear_ema.update(m.ear)
        mar = self._mar_ema.update(m.mar)
        pitch_rel = (m.pitch - self.pitch_offset) if m.pitch is not None else 0.0
        yaw_rel = (m.yaw - self.yaw_offset) if m.yaw is not None else 0.0
        self._pitch_rel, self._yaw_rel = pitch_rel, yaw_rel

        # ---------------- eyes (only when EAR is trustworthy)
        eye_valid = abs(yaw_rel) <= self.e.max_yaw_for_eye_analysis
        if eye_valid:
            if not self._eyes_closed and ear < self.ear_threshold:
                self._eyes_closed = True
                self._closure = 0.0
            elif self._eyes_closed and ear > self.ear_threshold + self.e.ear_hysteresis:
                self._end_eye_episode(t, events)
                self._eyes_closed = False
                self._closure = 0.0
                self._eyes_open = 0.0
            if self._eyes_closed:
                self._closure += dt
            else:
                self._eyes_open += dt
            self._eye_samples.append((t, dt, self._eyes_closed))
            self._eye_observed += dt
        else:
            # looking sideways: discard an ongoing closure instead of counting it
            self._eyes_closed = False
            self._closure = 0.0
            self._eyes_open = 0.0

        # ---------------- mouth / yawning
        if mar > self.mar_threshold:
            self._mouth_open += dt
            if not self._yawning and self._mouth_open >= self.m.yawn_min_duration:
                self._yawning = True
                self._yawns.append(t)
                self.total_yawns += 1
                events.append("YAWN STARTED")
        elif mar < self.mar_threshold - self.m.mar_hysteresis:
            if self._yawning:
                events.append(f"YAWN ENDED ({self._mouth_open:.1f}s)")
            self._yawning = False
            self._mouth_open = 0.0

        # ---------------- head pitch (nod / head drop)
        if self.h.enabled and m.pitch is not None:
            if pitch_rel > self.h.pitch_down_threshold:
                self._head_down_t += dt
                if not self._head_down and self._head_down_t >= self.h.nod_min_duration:
                    self._head_down = True
                    self._nods.append(t)
                    self.total_nods += 1
                    events.append(f"HEAD DROP (pitch {pitch_rel:+.0f} deg)")
            elif pitch_rel < self.h.pitch_down_threshold - self.h.pitch_hysteresis:
                if self._head_down:
                    events.append(f"HEAD UP ({self._head_down_t:.1f}s)")
                self._head_down = False
                self._head_down_t = 0.0

        return self._build_state(t, True, ear, mar, eye_valid, events)

    # ------------------------------------------------------------------ state snapshot
    def _build_state(self, t, face_visible, ear, mar, eye_valid, events) -> TemporalState:
        e = self.e
        self._prune(self._blinks, t, e.blink_rate_window)
        self._prune(self._long_blinks, t, e.perclos_window)
        self._prune(self._yawns, t, self.m.yawn_window)
        self._prune(self._nods, t, self.h.nod_window)
        while self._eye_samples and self._eye_samples[0][0] < t - e.perclos_window:
            self._eye_samples.popleft()

        total = sum(s[1] for s in self._eye_samples)
        closed = sum(s[1] for s in self._eye_samples if s[2])
        # denominator at least half a window: a single closure right after start-up must
        # not look like a huge PERCLOS (e.g. 1.5 s closed out of 3 s observed = 50 %)
        perclos = closed / max(total, e.perclos_window / 2)

        # blink rate only once at least half a window of valid eye data exists
        blink_rate = None
        if self._eye_observed >= e.blink_rate_window / 2:
            span = min(e.blink_rate_window, self._eye_observed)
            blink_rate = len(self._blinks) * 60.0 / span

        prev = self._last_state
        s = TemporalState(
            time=t,
            face_visible=face_visible,
            face_lost_duration=self._face_lost,
            ear=ear, mar=mar,
            pitch_rel=self._pitch_rel if ear is not None else prev.pitch_rel,
            yaw_rel=self._yaw_rel if ear is not None else prev.yaw_rel,
            eye_analysis_valid=eye_valid,
            eyes_closed=self._eyes_closed,
            closure_duration=self._closure,
            eyes_open_duration=self._eyes_open,
            recent_blink=t <= self._blink_shown_until,
            blink_rate=blink_rate,
            blinks_in_window=len(self._blinks),
            long_blinks_in_window=len(self._long_blinks),
            perclos=perclos,
            yawning=self._yawning,
            mouth_open_duration=self._mouth_open,
            yawns_in_window=len(self._yawns),
            head_down=self._head_down,
            head_down_duration=self._head_down_t if self._head_down else 0.0,
            nods_in_window=len(self._nods),
            looking_away=(ear is not None and abs(self._yaw_rel) > self.h.yaw_distraction_threshold),
            total_blinks=self.total_blinks,
            total_yawns=self.total_yawns,
            total_nods=self.total_nods,
            events=events,
        )
        self._last_state = s
        return s
