#!/usr/bin/env python3
"""
Alarm hardware check: switches the configured alarm backends on for 3 s, then off.

    python tools/test_alarm.py                     # backends from config.yaml
    python tools/test_alarm.py --alarm audio
    python tools/test_alarm.py --alarm gpio        # Raspberry Pi buzzer / relay
    python tools/test_alarm.py --alarm serial      # Arduino
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alarm import AlarmManager   # noqa: E402
from src.config import load_config   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--alarm", help="comma-separated backends")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = args.alarm.split(",") if args.alarm else None
    alarm = AlarmManager(cfg.alarm, names)
    print(f"Alarm ON for {args.seconds:.0f} s ...")
    alarm.on()
    time.sleep(args.seconds)
    alarm.off()
    print("Alarm OFF")
    alarm.close()


if __name__ == "__main__":
    main()
