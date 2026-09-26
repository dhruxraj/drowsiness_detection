"""Simulated alarm: prints to the console. Always available, used when no hardware exists."""
from __future__ import annotations

from .base import AlarmBackend


class SimulatedAlarm(AlarmBackend):
    name = "simulated"

    def __init__(self, cfg=None):
        self.active = False

    def on(self) -> None:
        if not self.active:
            self.active = True
            print("\a[SIMULATED ALARM] >>> BUZZER ON  - DROWSINESS DETECTED - TAKE A BREAK <<<", flush=True)

    def off(self) -> None:
        if self.active:
            self.active = False
            print("[SIMULATED ALARM] buzzer off", flush=True)
