"""Small helpers for time intervals given as ``(start_s, end_s)`` tuples in seconds."""
from __future__ import annotations

from typing import Iterable, List, Tuple

Interval = Tuple[float, float]


def normalize(intervals: Iterable[Interval]) -> List[Interval]:
    """Sort, drop empty intervals and merge overlapping/touching ones."""
    items = sorted((float(s), float(e)) for s, e in intervals if e > s)
    merged: List[Interval] = []
    for s, e in items:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def total_length(intervals: Iterable[Interval]) -> float:
    return sum(e - s for s, e in normalize(intervals))


def clip(intervals: Iterable[Interval], lo: float, hi: float) -> List[Interval]:
    return normalize((max(s, lo), min(e, hi)) for s, e in intervals)


def subtract(base: Iterable[Interval], remove: Iterable[Interval]) -> List[Interval]:
    """Return ``base`` minus the union of ``remove``."""
    result = normalize(base)
    for rs, re_ in normalize(remove):
        nxt: List[Interval] = []
        for s, e in result:
            if re_ <= s or rs >= e:
                nxt.append((s, e))
                continue
            if rs > s:
                nxt.append((s, rs))
            if re_ < e:
                nxt.append((re_, e))
        result = nxt
    return result


def contains(intervals: Iterable[Interval], t: float) -> bool:
    """True if ``t`` lies inside any interval (closed on both ends)."""
    return any(s <= t <= e for s, e in intervals)


def expand(interval: Interval, before: float, after: float) -> Interval:
    return (interval[0] - before, interval[1] + after)
