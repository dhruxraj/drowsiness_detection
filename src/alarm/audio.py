"""
Audible alarm through the computer speaker.

Tries, in order:
  1. pygame   - generated beep tone, looped (Windows / Linux / macOS / Raspberry Pi)
  2. winsound - Windows built-in fallback
  3. terminal bell - last resort
The tone is generated with NumPy, so no sound file is needed.
"""
from __future__ import annotations

import sys
import threading

import numpy as np

from .base import AlarmBackend


class AudioAlarm(AlarmBackend):
    name = "audio"

    def __init__(self, cfg):
        self.freq = float(cfg.frequency)
        self.beep_on = float(cfg.beep_on)
        self.beep_off = float(cfg.beep_off)
        self.volume = float(cfg.volume)
        self.active = False
        self._sound = None
        self._thread = None
        self._stop = threading.Event()
        self.method = self._init_pygame() or ("winsound" if sys.platform.startswith("win") else "bell")
        print(f"[audio alarm] using {self.method}")

    def _init_pygame(self) -> str | None:
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
            rate, _size, channels = pygame.mixer.get_init()
            t = np.arange(int(rate * self.beep_on)) / rate
            tone = np.sin(2 * np.pi * self.freq * t)
            fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)   # 10 ms fade -> no clicks
            tone = tone * fade * self.volume
            silence = np.zeros(int(rate * self.beep_off))
            cycle = (np.concatenate([tone, silence]) * 32767).astype(np.int16)
            if channels > 1:
                cycle = np.repeat(cycle[:, None], channels, axis=1)
            self._sound = pygame.mixer.Sound(buffer=np.ascontiguousarray(cycle).tobytes())
            return "pygame"
        except Exception as exc:  # pygame missing or no audio device
            print(f"[audio alarm] pygame unavailable ({exc}); falling back")
            return None

    def _fallback_loop(self) -> None:
        if self.method == "winsound":
            import winsound
            while not self._stop.is_set():
                winsound.Beep(int(self.freq), int(self.beep_on * 1000))
                self._stop.wait(self.beep_off)
        else:
            while not self._stop.is_set():
                sys.stdout.write("\a")
                sys.stdout.flush()
                self._stop.wait(self.beep_on + self.beep_off)

    def on(self) -> None:
        if self.active:
            return
        self.active = True
        if self._sound is not None:
            self._sound.play(loops=-1)
        else:
            self._stop.clear()
            self._thread = threading.Thread(target=self._fallback_loop, daemon=True)
            self._thread.start()

    def off(self) -> None:
        if not self.active:
            return
        self.active = False
        if self._sound is not None:
            self._sound.stop()
        else:
            self._stop.set()
