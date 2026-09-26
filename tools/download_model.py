#!/usr/bin/env python3
"""
Download the MediaPipe Face Landmarker model (needed only for face_mesh.engine: tasks,
i.e. MediaPipe versions that no longer ship mp.solutions).

    python tools/download_model.py
"""
import os
import sys
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "models", "face_landmarker.task")


def main() -> int:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.isfile(DEST) and os.path.getsize(DEST) > 1_000_000:
        print(f"Model already present: {DEST}")
        return 0
    print(f"Downloading {URL}\n -> {DEST}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:
        print(f"Download failed: {exc}\nDownload the file manually from the URL above "
              f"and save it as {DEST}")
        return 1
    print(f"Done ({os.path.getsize(DEST) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
