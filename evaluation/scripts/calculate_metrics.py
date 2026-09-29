#!/usr/bin/env python3
"""Recompute metrics and reports from a saved run (no MediaPipe/OpenCV needed).

    # recompute a run (e.g. after fixing a label) and write outputs into the run directory
    python evaluation/scripts/calculate_metrics.py --run-dir evaluation/results/runs/<run_id>

    # ... and publish it to evaluation/results/ if it is an official final result
    python evaluation/scripts/calculate_metrics.py --run-dir evaluation/results/runs/<run_id> --publish

    # (re)write evaluation/results/ stating that nothing has been measured yet
    python evaluation/scripts/calculate_metrics.py --placeholder

Note: matching tolerances are part of the frozen protocol. Changing them after looking at
final results and recomputing invalidates the evaluation (the frozen check will fail).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.core import eval_config, thresholds as th, workflow  # noqa: E402
from evaluation.core.dataset import DatasetError  # noqa: E402
from evaluation.core.pipeline import read_config_dict  # noqa: E402
from evaluation.core.provenance import utc_now  # noqa: E402
from evaluation.core.splits import assert_separated, load_split  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="run directory produced by evaluate.py")
    g.add_argument("--placeholder", action="store_true",
                   help="write 'not yet measured' results to evaluation/results/")
    p.add_argument("--config", default=str(eval_config.DEFAULT_EVAL_CONFIG))
    p.add_argument("--system-config")
    p.add_argument("--publish", action="store_true", help="copy an official final result to evaluation/results/")
    args = p.parse_args(argv)

    cfg = eval_config.load(args.config)
    sys_cfg_path = Path(args.system_config) if args.system_config else eval_config.resolve(cfg["paths"]["system_config"])
    system_cfg = read_config_dict(sys_cfg_path)
    frozen = workflow.load_frozen(eval_config.resolve(cfg["paths"]["frozen_thresholds"]))
    results_dir = eval_config.resolve(cfg["paths"]["results_dir"])

    try:
        tuning = load_split(cfg, "tuning", with_labels=False)
        final = load_split(cfg, "final")
        assert_separated(tuning, final)
    except DatasetError as exc:
        print("DATASET ERROR:\n  " + "\n  ".join(exc.errors))
        return 2

    if args.placeholder:
        workflow.write_placeholder(results_dir, final, system_cfg, cfg, frozen)
        print(f"Wrote 'not yet measured' results to {results_dir}")
        return 0

    run_dir = Path(args.run_dir)
    meta, runs = workflow.load_run(run_dir)
    split = final if meta["split"] == "final" else load_split(cfg, "tuning")
    ids = set(meta["recordings"])
    evaluated_split = dataclasses.replace(split, recordings=[r for r in split.recordings if r.recording_id in ids])
    frozen_problems = []
    if meta["split"] == "final":
        frozen_problems = th.verify_frozen(frozen, system_cfg, cfg, [r.recording_id for r in split.recordings],
                                           [r.subject_id for r in split.recordings])
        if th.fingerprint(th.detection_snapshot(system_cfg)) != meta.get("detection_fingerprint"):
            frozen_problems.append("config.yaml changed since this run was produced")
    evals, summary, bd = workflow.evaluate_runs(evaluated_split, runs, cfg, system_cfg)
    official, why = workflow.official_status(meta, split, frozen_problems)
    context = {"generated_at": utc_now(), "split": meta["split"], "run_id": meta["run_id"], "git": meta.get("git"),
               "frame_step": meta.get("frame_step", 1),
               "environment": meta.get("environment"), "landmark_engine": meta.get("landmark_engine"),
               "official": official, "not_official_reason": why}
    workflow.write_outputs(run_dir, context=context, split=evaluated_split, summary=summary, bd=bd,
                           system_cfg=system_cfg, eval_cfg=cfg, frozen=frozen)
    workflow.write_event_tables(run_dir, evals)
    print(f"Metrics written to {run_dir}")
    if args.publish:
        if not official:
            print(f"Not published - not an official final result: {why}")
            return 3
        workflow.publish(run_dir, results_dir)
        print(f"Published to {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
