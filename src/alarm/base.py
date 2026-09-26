"""
Alarm backend interface.

Every output device (speaker, GPIO buzzer, relay, Arduino, vehicle alarm ...) implements
the same three methods, so new hardware can be added without touching the detection code:

    class MyVehicleAlarm(AlarmBackend):
        name = "vehicle"
        def on(self):  ...   # start warning
        def off(self): ...   # stop warning
        def close(self): ... # release hardware

Then register it in alarm/manager.py -> BACKENDS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class AlarmBackend(ABC):
    name = "base"

    @abstractmethod
    def on(self) -> None:
        """Start the alarm. Must return immediately (use a thread for beep patterns)."""

    @abstractmethod
    def off(self) -> None:
        """Stop the alarm."""

    def close(self) -> None:
        """Release resources (called once at shutdown)."""
        self.off()
