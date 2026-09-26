"""
Arduino (or any microcontroller) alarm over USB serial.

Protocol (one ASCII byte):  'A' = alarm on,  'S' = alarm off
See hardware/arduino_buzzer/arduino_buzzer.ino for the matching sketch.
"""
from __future__ import annotations

import threading
import time

from .base import AlarmBackend


class SerialAlarm(AlarmBackend):
    name = "serial"

    def __init__(self, cfg):
        import serial  # pyserial
        self.ser = serial.Serial(cfg.port, int(cfg.baudrate), timeout=0.1)
        time.sleep(2.0)                   # most Arduinos reset when the port opens
        self.active = False
        self._lock = threading.Lock()
        self._send(b"S")

    def _send(self, b: bytes) -> None:
        with self._lock:
            try:
                self.ser.write(b)
                self.ser.flush()
            except Exception as exc:
                print(f"[serial alarm] write failed: {exc}")

    def on(self) -> None:
        if not self.active:
            self.active = True
            self._send(b"A")

    def off(self) -> None:
        if self.active:
            self.active = False
            self._send(b"S")

    def close(self) -> None:
        self._send(b"S")
        self.ser.close()
