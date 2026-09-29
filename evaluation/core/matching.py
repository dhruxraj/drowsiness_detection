"""Event-level matching of predicted alarms against ground-truth events.

The system produces *temporal* alarms (an alarm turns on, stays on for at least
``alarm.min_duration`` and turns off with hysteresis), so matching is done on alarm
**episodes**, not on individual frames.

Rule (documented in evaluation/README.md, "Event matching"):

* A predicted alarm episode is described by its onset time (OFF -> ON transition).
* Each alarm-expected ground-truth event ``[start, end]`` gets a matching window
  ``[start - pre_tolerance_s, end + post_tolerance_s]``.
* Onsets are processed in time order. An onset inside the window of a not-yet-matched
  event is a **detection** of that event (earliest-starting eligible event first; one
  onset can detect at most one event, one event is detected at most once).
* An onset inside the window of an event that is already detected is a **duplicate**
  (reported separately, not a false alarm).
* Otherwise, an onset inside an ``alarm_expected=ignore`` window (same tolerances) is
  **ignored**.
* Any remaining onset inside the evaluation window is a **false alarm**.
* An alarm-expected event without a detection is a **missed event**.
* Onsets before the evaluation window (calibration period) are counted separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .dataset import LabelEvent
from .trace import FrameRecord


@dataclass(frozen=True)
class AlarmEpisode:
    onset_s: float
    offset_s: Optional[float]  # None if the alarm was still on at the end of the recording


@dataclass
class Detection:
    event: LabelEvent
    episode: AlarmEpisode

    @property
    def latency_ms(self) -> float:
        return (self.episode.onset_s - self.event.start_s) * 1000.0

    @property
    def release_delay_ms(self) -> Optional[float]:
        if self.episode.offset_s is None:
            return None
        return (self.episode.offset_s - self.event.end_s) * 1000.0


@dataclass
class MatchResult:
    detections: List[Detection] = field(default_factory=list)
    missed: List[LabelEvent] = field(default_factory=list)
    false_alarms: List[AlarmEpisode] = field(default_factory=list)
    duplicates: List[AlarmEpisode] = field(default_factory=list)
    ignored: List[AlarmEpisode] = field(default_factory=list)
    before_window: List[AlarmEpisode] = field(default_factory=list)


def extract_episodes(frames: Sequence[FrameRecord]) -> List[AlarmEpisode]:
    """Alarm episodes from a per-frame trace (onset = first frame with alarm_on)."""
    episodes: List[AlarmEpisode] = []
    onset: Optional[float] = None
    prev = False
    for f in frames:
        if f.alarm_on and not prev:
            onset = f.t_s
        elif prev and not f.alarm_on:
            episodes.append(AlarmEpisode(onset, f.t_s))
            onset = None
        prev = f.alarm_on
    if onset is not None:
        episodes.append(AlarmEpisode(onset, None))
    return episodes


def _window(e: LabelEvent, pre: float, post: float) -> Tuple[float, float]:
    return (e.start_s - pre, e.end_s + post)


def match_alarms(events: Sequence[LabelEvent], episodes: Sequence[AlarmEpisode],
                 eval_start_s: float, pre_tolerance_s: float, post_tolerance_s: float) -> MatchResult:
    result = MatchResult()
    positives = sorted((e for e in events if e.is_positive), key=lambda e: (e.start_s, e.event_id))
    ignores = [_window(e, pre_tolerance_s, post_tolerance_s) for e in events if e.is_ignore]
    matched = {}
    for ep in sorted(episodes, key=lambda x: x.onset_s):
        t = ep.onset_s
        if t < eval_start_s:
            result.before_window.append(ep)
            continue
        in_window = [e for e in positives
                     if _window(e, pre_tolerance_s, post_tolerance_s)[0] <= t
                     <= _window(e, pre_tolerance_s, post_tolerance_s)[1]]
        free = [e for e in in_window if e.event_id not in matched]
        if free:
            target = free[0]
            matched[target.event_id] = Detection(target, ep)
        elif in_window:
            result.duplicates.append(ep)
        elif any(s <= t <= e for s, e in ignores):
            result.ignored.append(ep)
        else:
            result.false_alarms.append(ep)
    for e in positives:
        if e.event_id in matched:
            result.detections.append(matched[e.event_id])
        else:
            result.missed.append(e)
    return result
