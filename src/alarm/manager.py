"""
AlarmManager - drives all configured alarm backends together.

* A backend that fails to initialise (e.g. no GPIO on a laptop) is skipped with a warning.
* The simulated backend is always added as a fallback, so the system never runs "silently".
* test() sounds the alarm briefly to check the hardware.
"""
from __future__ import annotations

import threading

from .base import AlarmBackend
from .simulated import SimulatedAlarm


def _make(name: str, cfg):
    if name == "simulated":
        return SimulatedAlarm()
    if name == "audio":
        from .audio import AudioAlarm
        return AudioAlarm(cfg.audio)
    if name == "gpio":
        from .gpio import GPIOAlarm
        return GPIOAlarm(cfg.gpio)
    if name == "serial":
        from .serial_alarm import SerialAlarm
        return SerialAlarm(cfg.serial)
    raise ValueError(f"unknown alarm backend '{name}'")


class AlarmManager:
    def __init__(self, cfg, backend_names: list[str] | None = None):
        self.backends: list[AlarmBackend] = []
        for name in backend_names or cfg.backends:
            try:
                self.backends.append(_make(str(name).lower(), cfg))
            except Exception as exc:
                print(f"[alarm] backend '{name}' not available: {exc}")
        if not any(isinstance(b, SimulatedAlarm) for b in self.backends):
            self.backends.append(SimulatedAlarm())
        self.active = False
        self._testing = False
        print("[alarm] active backends: " + ", ".join(b.name for b in self.backends))

    def _each(self, method: str) -> None:
        for b in self.backends:
            try:
                getattr(b, method)()
            except Exception as exc:
                print(f"[alarm] {b.name}.{method}() failed: {exc}")

    def on(self) -> None:
        self._testing = False
        if not self.active:
            self.active = True
            self._each("on")

    def off(self) -> None:
        self._testing = False
        if self.active:
            self.active = False
            self._each("off")

    def test(self, duration: float = 1.5) -> None:
        """Sound the alarm briefly (does not interfere with a real alarm)."""
        if self.active:
            return
        self.active, self._testing = True, True
        self._each("on")

        def _end():
            if self._testing:
                self.off()
        threading.Timer(duration, _end).start()

    def close(self) -> None:
        self.off()
        self._each("close")
