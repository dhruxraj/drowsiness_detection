#!/usr/bin/env python3
"""Validate manifests and labels of both splits and their separation.

    python evaluation/scripts/validate_dataset.py                 # schema + separation
    python evaluation/scripts/validate_dataset.py --check-files   # + video files exist
    python evaluation/scripts/validate_dataset.py --write-hashes  # fill empty sha256 cells, verify others
    python evaluation/scripts/validate_dataset.py --probe         # compare fps/resolution/duration with the video (OpenCV)

Exit code 0 = valid, 2 = errors found.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.core import eval_config  # noqa: E402
from evaluation.core.dataset import DatasetError  # noqa: E402
from evaluation.core.metrics import dataset_summary  # noqa: E402
from evaluation.core.provenance import sha256_file  # noqa: E402
from evaluation.core.splits import check_separation, load_split  # noqa: E402


def write_hashes(split) -> list:
    """Fill empty sha256 cells; report mismatching existing hashes. Returns error list."""
    errors = []
    path = split.recordings_path
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "sha256" not in fieldnames:
        fieldnames.insert(fieldnames.index("file") + 1, "sha256")
    changed = False
    for row in rows:
        video = split.dataset_dir / row["file"].strip()
        if not video.is_file():
            errors.append(f"{split.name}/{row['recording_id']}: file not found: {video}")
            continue
        digest = sha256_file(video)
        current = (row.get("sha256") or "").strip().lower()
        if not current:
            row["sha256"] = digest
            changed = True
        elif current != digest:
            errors.append(f"{split.name}/{row['recording_id']}: sha256 mismatch - the file changed "
                          "after it was registered (labels may no longer fit)")
    if changed:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"updated sha256 values in {path}")
    return errors


def probe(split) -> list:
    try:
        import cv2
    except ImportError:
        return [f"--probe needs OpenCV (pip install -r requirements.txt)"]
    notes = []
    for rec in split.recordings:
        cap = cv2.VideoCapture(str(split.video_path(rec)))
        if not cap.isOpened():
            notes.append(f"ERROR {rec.recording_id}: OpenCV cannot open the file")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        dur = n / fps if fps > 0 else 0.0
        if fps and abs(fps - rec.source_fps) > 0.5:
            notes.append(f"{rec.recording_id}: manifest source_fps={rec.source_fps} but video reports {fps:.2f}")
        if (w, h) != (rec.width, rec.height):
            notes.append(f"{rec.recording_id}: manifest resolution={rec.resolution} but video is {w}x{h}")
        if dur and abs(dur - rec.duration_s) > max(0.5, 2.0 / max(fps, 1.0)):
            notes.append(f"{rec.recording_id}: manifest duration_s={rec.duration_s} but video reports {dur:.2f}s")
    return notes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(eval_config.DEFAULT_EVAL_CONFIG))
    p.add_argument("--check-files", action="store_true")
    p.add_argument("--write-hashes", action="store_true")
    p.add_argument("--probe", action="store_true")
    args = p.parse_args(argv)
    cfg = eval_config.load(args.config)

    errors, splits = [], {}
    for name in ("tuning", "final"):
        try:
            splits[name] = load_split(cfg, name, check_files=args.check_files or args.write_hashes)
        except DatasetError as exc:
            errors += [f"[{name}] {e}" for e in exc.errors]
    if errors:
        print("INVALID:\n  " + "\n  ".join(errors))
        return 2
    if args.write_hashes:
        for s in splits.values():
            errors += write_hashes(s)
        if errors:
            print("INVALID:\n  " + "\n  ".join(errors))
            return 2
        splits = {n: load_split(cfg, n) for n in splits}
    problems = check_separation(splits["tuning"], splits["final"])
    if problems:
        print("TUNING/FINAL LEAKAGE:\n  " + "\n  ".join(problems))
        return 2
    for s in splits.values():
        ds = dataset_summary(s.recordings, s.labels)
        print(f"[{s.name}] subjects={ds['subjects']} recordings={ds['recordings']} "
              f"duration={ds['total_duration_s'] / 60:.1f} min events="
              f"{sum(v['count'] for v in ds['event_counts'].values())}")
        for w in s.warnings:
            print(f"  warning: {w}")
        if args.probe:
            for note in probe(s):
                print(f"  probe: {note}")
    if not any(s.recordings for s in splits.values()):
        print("No recordings are registered yet (the manifests only contain the header row). "
              "See evaluation/dataset/README.md.")
    print("OK - labels valid and tuning/final splits are separated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
