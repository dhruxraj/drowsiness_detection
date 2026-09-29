#!/usr/bin/env python3
"""Run labelled recordings through the production pipeline and compute the metrics.

Examples (run from the repository root):

    # tuning split (any time; results stay in evaluation/results/runs/)
    python evaluation/scripts/evaluate.py --split tuning

    # frame-rate robustness: same recordings, every 2nd frame (never an official result)
    python evaluation/scripts/evaluate.py --split tuning --frame-step 2

    # final split (requires frozen thresholds; publishes to evaluation/results/)
    python evaluation/scripts/evaluate.py --split final \
        --dataset evaluation/dataset/final \
        --recordings evaluation/labels/evaluation_recordings.csv \
        --labels evaluation/labels/evaluation_labels.csv \
        --config evaluation/config/evaluation_config.yaml

Exit codes: 0 ok, 1 nothing to evaluate / missing dependency, 2 invalid dataset or
tuning/final leakage, 3 thresholds not frozen or changed since freezing.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.core import eval_config, thresholds as th, workflow  # noqa: E402
from evaluation.core.dataset import DatasetError  # noqa: E402
from evaluation.core.pipeline import read_config_dict  # noqa: E402
from evaluation.core.provenance import environment, git_info, utc_now  # noqa: E402
from evaluation.core.splits import assert_separated, load_split  # noqa: E402
from evaluation.core.trace import write_trace  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=("tuning", "final"))
    p.add_argument("--config", default=str(eval_config.DEFAULT_EVAL_CONFIG), help="evaluation config YAML")
    p.add_argument("--system-config", help="detection config (default: paths.system_config, i.e. config.yaml)")
    p.add_argument("--dataset", help="directory containing this split's video files")
    p.add_argument("--recordings", help="recordings manifest CSV for this split")
    p.add_argument("--labels", help="labels CSV for this split")
    p.add_argument("--out", help="run directory (default: evaluation/results/runs/<timestamp>_<split>)")
    p.add_argument("--only", nargs="+", metavar="RECORDING_ID",
                   help="process only these recordings (debugging; never an official result)")
    p.add_argument("--allow-unfrozen", action="store_true",
                   help="final split only: run even if thresholds are not frozen. The result is "
                        "marked as NOT official and is not published.")
    p.add_argument("--frame-step", type=int, default=1, metavar="N",
                   help="process only every N-th frame (timestamps stay on the video clock) to check that "
                        "the time-based analysis is robust to a lower processing frame rate; such runs are "
                        "never official final results (default: 1 = every frame)")
    p.add_argument("--no-publish", action="store_true", help="do not copy official results to evaluation/results/")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.frame_step < 1:
        print("--frame-step must be >= 1")
        return 2
    cfg = eval_config.load(args.config)
    sys_cfg_path = Path(args.system_config) if args.system_config else eval_config.resolve(cfg["paths"]["system_config"])
    other = "final" if args.split == "tuning" else "tuning"

    # ---- dataset + split separation
    try:
        split = load_split(cfg, args.split, check_files=True, dataset_dir=args.dataset,
                           recordings_path=args.recordings, labels_path=args.labels)
        other_split = load_split(cfg, other, with_labels=False)
        assert_separated(split if args.split == "tuning" else other_split,
                         split if args.split == "final" else other_split)
    except DatasetError as exc:
        print("DATASET ERROR - nothing was evaluated:\n  " + "\n  ".join(exc.errors))
        return 2
    for w in split.warnings:
        print(f"warning: {w}")

    system_cfg = read_config_dict(sys_cfg_path)
    frozen = workflow.load_frozen(eval_config.resolve(cfg["paths"]["frozen_thresholds"]))
    frozen_problems = []
    if args.split == "final":
        frozen_problems = th.verify_frozen(frozen, system_cfg, cfg,
                                           [r.recording_id for r in split.recordings],
                                           [r.subject_id for r in split.recordings])
        if frozen_problems and not args.allow_unfrozen:
            print("REFUSING to evaluate the final split:\n  " + "\n  ".join(frozen_problems))
            return 3

    recordings = split.recordings
    if args.only:
        unknown = sorted(set(args.only) - {r.recording_id for r in recordings})
        if unknown:
            print(f"unknown recording id(s) for --only: {unknown}")
            return 2
        recordings = [r for r in recordings if r.recording_id in set(args.only)]
    if not recordings:
        print(f"No recordings in {split.recordings_path}. Nothing to evaluate - see "
              "evaluation/dataset/README.md for how to record and label data.")
        return 1

    try:
        from evaluation.core.pipeline import ProductionPipeline, load_production_config
        from evaluation.core.runner import CameraFrameSource, run_recording
        load_production_config(sys_cfg_path)  # fail early on an invalid config
    except ImportError as exc:
        print(f"Cannot import the production pipeline ({exc}). Install requirements.txt first.")
        return 1

    stamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
    base_id = f"{stamp}_{args.split}" + (f"_step{args.frame_step}" if args.frame_step != 1 else "")
    if args.out:
        run_dir = Path(args.out)
        if (run_dir / workflow.RUN_META).exists():
            print(f"{run_dir} already contains a run; choose another --out directory.")
            return 2
        run_id = run_dir.name
    else:
        runs_root = eval_config.resolve(cfg["paths"]["runs_dir"])
        run_id, n = base_id, 1
        while (runs_root / run_id).exists():  # never overwrite an earlier run
            n += 1
            run_id = f"{base_id}-{n}"
        run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id, "split": args.split, "created_at": utc_now(), "git": git_info(),
        "environment": environment(), "system_config": str(sys_cfg_path),
        "detection_fingerprint": th.fingerprint(th.detection_snapshot(system_cfg)),
        "protocol_fingerprint": th.fingerprint(th.protocol_snapshot(cfg)),
        "frozen_detection_fingerprint": (frozen or {}).get("detection_fingerprint"),
        "subset": bool(args.only), "frame_step": args.frame_step, "landmark_engine": None, "recordings": {},
    }
    runs = {}
    engine_names = set()
    for i, rec in enumerate(recordings, 1):
        video = split.video_path(rec)
        prod_cfg = load_production_config(sys_cfg_path)  # fresh config object per recording
        holder = {}

        def make_pipeline():
            holder["p"] = ProductionPipeline(prod_cfg)
            return holder["p"]

        def make_source():
            return CameraFrameSource(prod_cfg, video, cfg["runner"]["read_timeout_s"],
                                     cfg["runner"]["stall_timeout_s"])

        print(f"[{i}/{len(recordings)}] {rec.recording_id} ({rec.subject_id}, {rec.lighting}, "
              f"glasses={rec.glasses}) ...", flush=True)
        run = run_recording(rec.recording_id, make_pipeline, make_source, frame_step=args.frame_step)
        if "p" in holder:
            engine_names.add(holder["p"].engine_name)
        write_trace(run_dir / workflow.TRACES_DIR / f"{rec.recording_id}.csv", run.frames)
        meta["recordings"][rec.recording_id] = run.meta()
        runs[rec.recording_id] = run
        print(f"    status={run.status} frames={len(run.frames)} {run.error}".rstrip())
    meta["landmark_engine"] = ", ".join(sorted(e for e in engine_names if e)) or None
    (run_dir / workflow.RUN_META).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    evaluated_split = dataclasses.replace(split, recordings=recordings)
    evals, summary, bd = workflow.evaluate_runs(evaluated_split, runs, cfg, system_cfg)
    official, why = workflow.official_status(meta, split, frozen_problems)
    context = {"generated_at": utc_now(), "split": args.split, "run_id": run_id, "git": meta["git"],
               "frame_step": args.frame_step,
               "environment": meta["environment"], "landmark_engine": meta["landmark_engine"],
               "official": official, "not_official_reason": why}
    workflow.write_outputs(run_dir, context=context, split=evaluated_split, summary=summary, bd=bd,
                           system_cfg=system_cfg, eval_cfg=cfg, frozen=frozen)
    workflow.write_event_tables(run_dir, evals)
    print(f"\nRun directory: {run_dir}")
    print(f"  false alarms/h: {summary['false_alarms_per_hour']}  missed: {summary['missed_events']}/"
          f"{summary['positive_events']}  latency median ms: {summary['latency_ms']['median']}  "
          f"measurement loss %: {summary['measurement_unavailable_rate_pct']}  "
          f"processing fps: {summary['processing_fps_mean']}")
    for w in summary["warnings"]:
        print(f"warning: {w}")
    if official and not args.no_publish:
        workflow.publish(run_dir, eval_config.resolve(cfg["paths"]["results_dir"]))
        print("Official final results published to evaluation/results/.")
    elif not official:
        print(f"Not an official final result: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
