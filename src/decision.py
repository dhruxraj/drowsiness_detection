"""
Decision logic: detection state + alarm on/off with hysteresis.

Alarm ON   when  total score >= scoring.threshold
Alarm OFF  when  ALL of:
             * alarm has sounded for at least alarm.min_duration
             * the face is visible and the eyes have been open for alarm.release_time
             * the head is not dropped
             * acute score < scoring.release_threshold
The different ON/OFF conditions prevent the alarm from flickering on and off.
If the face disappears while the alarm is sounding, the alarm keeps sounding
(the driver may have slumped) until normal alertness is observed again.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .drowsiness_scorer import ScoreBreakdown
from .temporal_analyzer import TemporalState


class DetectionState(str, Enum):
    CALIBRATING = "CALIBRATING"
    NO_FACE = "NO FACE"
    ALERT = "ALERT"
    BLINKING = "BLINKING"
    EYES_CLOSED = "EYES CLOSED"
    YAWNING = "YAWNING"
    HEAD_DROP = "HEAD DROP"
    LOOKING_AWAY = "LOOKING AWAY"
    DROWSY = "DROWSY"


@dataclass
class Decision:
    state: DetectionState
    alarm_on: bool
    alarm_changed: bool = False
    fatigue_warning: bool = False
    driver_not_visible: bool = False


class DecisionEngine:
    def __init__(self, cfg):
        self.s, self.a, self.e, self.fl = cfg.scoring, cfg.alarm, cfg.eyes, cfg.face_loss
        self.reset()

    def reset(self) -> None:
        self.alarm_on = False
        self.alarm_started_at = 0.0
        self.alarm_count = 0
        self._state = DetectionState.ALERT

    def calibrating(self) -> Decision:
        return Decision(DetectionState.CALIBRATING, self.alarm_on)

    def _classify(self, st: TemporalState) -> DetectionState:
        if not st.face_visible:
            return DetectionState.NO_FACE
        if st.ear is None:                       # brief dropout inside grace period
            return self._state
        if st.head_down:
            return DetectionState.HEAD_DROP
        if st.eyes_closed and st.closure_duration > self.e.blink_max_duration:
            return DetectionState.EYES_CLOSED
        if st.yawning:
            return DetectionState.YAWNING
        if st.eyes_closed or st.recent_blink:
            return DetectionState.BLINKING
        if st.looking_away:
            return DetectionState.LOOKING_AWAY
        return DetectionState.ALERT

    def update(self, st: TemporalState, score: ScoreBreakdown, t: float) -> Decision:
        changed = False
        if not self.alarm_on:
            if score.total >= self.s.threshold:
                self.alarm_on, changed = True, True
                self.alarm_started_at = t
                self.alarm_count += 1
        else:
            recovered = (
                st.face_visible and st.ear is not None
                and st.eye_analysis_valid
                and not st.eyes_closed
                and st.eyes_open_duration >= self.a.release_time
                and not st.head_down
                and score.acute < self.s.release_threshold
            )
            if recovered and t - self.alarm_started_at >= self.a.min_duration:
                self.alarm_on, changed = False, True

        base = self._classify(st)
        self._state = base
        state = DetectionState.DROWSY if self.alarm_on else base
        return Decision(
            state=state,
            alarm_on=self.alarm_on,
            alarm_changed=changed,
            fatigue_warning=score.cumulative >= self.s.fatigue_warning,
            driver_not_visible=st.face_lost_duration >= self.fl.warn_after,
        )
