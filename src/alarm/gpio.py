"""
Raspberry Pi GPIO output: active buzzer or relay module (e.g. to drive a vehicle horn/alarm).

Uses gpiozero (pre-installed on Raspberry Pi OS). BCM pin numbering.
Wiring (active buzzer):  GPIO18 (pin 12) -> buzzer +,  GND (pin 6) -> buzzer -
For a buzzer drawing > ~15 mA or for a relay, drive it through a transistor or use a
relay MODULE that already contains the driver transistor and flyback diode.
"""
from __future__ import annotations

from .base import AlarmBackend


class GPIOAlarm(AlarmBackend):
    name = "gpio"

    def __init__(self, cfg):
        from gpiozero import Buzzer, OutputDevice   # raises ImportError on non-Pi machines
        self.device_type = str(cfg.device).lower()
        self.beep_on, self.beep_off = float(cfg.beep_on), float(cfg.beep_off)
        if self.device_type == "relay":
            self.dev = OutputDevice(int(cfg.pin), active_high=bool(cfg.active_high), initial_value=False)
        else:
            self.dev = Buzzer(int(cfg.pin), active_high=bool(cfg.active_high))
        self.active = False

    def on(self) -> None:
        if self.active:
            return
        self.active = True
        if self.device_type == "relay":
            self.dev.on()                       # relay: steady output
        else:
            self.dev.beep(on_time=self.beep_on, off_time=self.beep_off, background=True)

    def off(self) -> None:
        self.active = False
        self.dev.off()

    def close(self) -> None:
        self.off()
        self.dev.close()
