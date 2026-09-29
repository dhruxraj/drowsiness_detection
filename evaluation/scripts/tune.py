#!/usr/bin/env python3
"""Select thresholds on the TUNING split only.

Replays the per-frame measurements of a tuning run (EAR/MAR/pose, face presence) through
the production calibration / temporal-analysis / scoring / decision code for every
combination in ``tuning.search_space`` of evaluation_config.yaml, and ranks them:

  1. feasible = false alarms/h <= tuning.max_false_alarms_per_hour
  2. among feasible: fewest missed events, then fewest false alarms/h, then lowest
     median latency.

Before the search, the current config is replayed and must reproduce the recorded alarm
sequence exactly; otherwise the replay is not faithful and the script stops.

    python evaluation/scripts/evaluate.py --split tuning
    python evaluation/scripts/tune.py --run-dir evaluation/results/runs/<run_id>_tuning

The script never edits config.yaml. Apply the suggested values by hand, re-run the tuning
evaluation to confirm, then freeze with freeze_thresholds.py.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.core import eval_config, thresholds as th, workflow  # noqa: E402
from evaluation.core.dataset import DatasetError  # noqa: E402
from evaluation.core.pipeline import read_config_dict  # noqa: E402
from evaluation.core.splits import assert_separated, load_split  # noqa: E402
from evaluation.core.trace import EVALUATED_STATUSES, RecordingRun  # noqa: E402


def rank_key(row, max_fa):
    fa = row["false_alarms_per_hour"]
    feasible = fa is not None and fa <= max_fa
    lat = row["latency_median_ms"]
    return (0 if feasible else 1, row["missed_events"],
            fa if fa is not None else math.inf, lat if lat is not None else math.inf)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="a run directory produced by evaluate.py --split tuning")
    p.add_argument("--config", default=str(eval_config.DEFAULT_EVAL_CONFIG))
    p.add_argument("--system-config")
    args = p.parse_args(argv)

    cfg = eval_config.load(args.config)
    sys_cfg_path = Path(args.system_config) if args.system_config else eval_config.resolve(cfg["paths"]["system_config"])
    run_dir = Path(args.run_dir)
    meta, runs = workflow.load_run(run_dir)
    if meta.get("split") != "tuning":
        print("REFUSING: tune.py only accepts runs on the TUNING split.")
        return 2
    try:
        tuning = load_split(cfg, "tuning")
        final = load_split(cfg, "final", with_labels=False)
        assert_separated(tuning, final)
    except DatasetError as exc:
        print("DATASET ERROR:\n  " + "\n  ".join(exc.errors))
        return 2
    final_ids = {r.recording_id.lower() for r in final.recordings}
    if any(rid.lower() in final_ids for rid in runs):
        print("REFUSING: the run contains final-split recordings.")
        return 2
    space = cfg["tuning"]["search_space"]
    if not space:
        print("tuning.search_space in evaluation_config.yaml is empty - nothing to search.")
        return 1
    system_cfg = read_config_dict(sys_cfg_path)
    if th.fingerprint(th.detection_snapshot(system_cfg)) != meta.get("detection_fingerprint"):
        print("config.yaml changed since this tuning run was recorded; re-run "
              "evaluate.py --split tuning first so the baseline replay can be verified.")
        return 2

    from evaluation.core.pipeline import load_production_config, make_frame_metrics
    from evaluation.core.runner import replay_decisions

    usable = {rid: r for rid, r in runs.items() if r.status in EVALUATED_STATUSES}
    for rid, run in usable.items():
        replayed = replay_decisions(run.frames, load_production_config(sys_cfg_path), make_frame_metrics)
        if [f.alarm_on for f in replayed] != [f.alarm_on for f in run.frames]:
            print(f"Replay of {rid} with the unchanged config does not reproduce the recorded alarms; "
                  "trace replay is not faithful for this pipeline version. Aborting.")
            return 2

    keys = list(space)
    rows = []
    sub = dataclasses.replace(tuning, recordings=[r for r in tuning.recordings if r.recording_id in runs])
    for combo in itertools.product(*(space[k] for k in keys)):
        overrides = dict(zip(keys, combo))
        replayed_runs = {}
        for rid, run in runs.items():
            if run.status in EVALUATED_STATUSES:
                cand_cfg = load_production_config(sys_cfg_path, overrides)
                frames = replay_decisions(run.frames, cand_cfg, make_frame_metrics)
            else:
                frames = run.frames
            replayed_runs[rid] = RecordingRun(rid, run.status, run.error, run.startup_s, "", frames,
                                              frame_step=run.frame_step)
        _, s, _ = workflow.evaluate_runs(sub, replayed_runs, cfg, system_cfg)
        rows.append({**overrides, "positive_events": s["positive_events"], "missed_events": s["missed_events"],
                     "false_alarms": s["false_alarms"], "false_alarms_per_hour": s["false_alarms_per_hour"],
                     "latency_median_ms": s["latency_ms"]["median"]})
        print(f"{overrides} -> missed {s['missed_events']}/{s['positive_events']}, "
              f"FA/h {s['false_alarms_per_hour']}, latency median {s['latency_ms']['median']}")

    max_fa = float(cfg["tuning"]["max_false_alarms_per_hour"])
    rows.sort(key=lambda r: rank_key(r, max_fa))
    out_dir = run_dir / "tuning"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "tuning_results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = rows[0]
    feasible = rank_key(best, max_fa)[0] == 0
    import yaml
    (out_dir / "best_overrides.yaml").write_text(
        "# Suggested by tune.py on the TUNING split only. Apply to config.yaml by hand, re-run the\n"
        "# tuning evaluation to confirm, then run freeze_thresholds.py.\n"
        f"# feasible (FA/h <= {max_fa}): {feasible}\n" + yaml.safe_dump({k: best[k] for k in keys}, sort_keys=False),
        encoding="utf-8")
    print(f"\nBest candidate ({'feasible' if feasible else 'NO candidate met the false-alarm limit'}): "
          f"{ {k: best[k] for k in keys} }")
    print(f"Results: {out_dir / 'tuning_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
