#!/usr/bin/env python3
"""
Camera check: lists working camera indices, then shows the selected camera with its
actual resolution, FPS and brightness.  q = quit

    python tools/test_camera.py            # scan + open camera 0
    python tools/test_camera.py --index 1
"""
import argparse
import time

import cv2


def scan(max_index: int = 5):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            found.append(i)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    print("Working camera indices:", scan() or "none found")

    cap = cv2.VideoCapture(args.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Cannot open camera {args.index}")
        return
    last, fps = time.perf_counter(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed")
            break
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        bright = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        h, w = frame.shape[:2]
        txt = f"{w}x{h}  {fps:4.1f} FPS  brightness {bright:5.1f}"
        cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if bright < 70:
            cv2.putText(frame, "TOO DARK - add light / IR illumination", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("camera test (q = quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
