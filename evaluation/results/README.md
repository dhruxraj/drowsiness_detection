# Evaluation results

> **Current status: Not yet measured — real labelled recordings required.**
> No real recordings have been evaluated. `metrics.csv`, `breakdown.csv` and `report.md`
> in this folder were generated with
> `python evaluation/scripts/calculate_metrics.py --placeholder` and state this
> explicitly; they contain no experimental values.

## Files

| File | Content |
|---|---|
| `metrics.csv` | overall metrics, machine-readable |
| `breakdown.csv` | the same metrics per subject and per condition (lighting, glasses, head position, camera distance) |
| `report.md` | human-readable report: dataset, protocol, frozen thresholds, results, breakdowns, environment, limitations |
| `summary.json` | every computed quantity (written only for a measured result) |
| `runs/` | one directory per evaluation run with per-frame traces - **git-ignored**, never committed |

Only an **official** result is written here: a run of the complete FINAL split with frozen,
unchanged thresholds (`evaluate.py --split final`, or
`calculate_metrics.py --run-dir <run> --publish`). Tuning-split runs, partial runs
(`--only`) and runs with `--allow-unfrozen` stay in `runs/<run_id>/` and are marked
"NOT an official final result".

## CSV columns

`scope, group, metric, value, unit, sample_size, status`

* `scope` / `group` - `overall` / `all` in `metrics.csv`; the attribute and its value
  (e.g. `lighting` / `dim_indoor`) in `breakdown.csv`.
* `metric` - e.g. `false_alarms_per_hour`, `missed_events`, `latency_ms_median`,
  `measurement_unavailable_rate_pct`, `processing_fps_mean` (definitions:
  [`../README.md`](../README.md#metric-definitions)).
* `unit` - `1/h`, `count`, `%`, `% of frames`, `ms`, `frames/s`.
* `sample_size` - what the value is based on (non-event hours, alarm-expected events,
  detected events, evaluated frames, processed frames, recordings).
* `status` - `measured`, `undefined` (e.g. no detected events, so no latency),
  `insufficient_data` (breakdown group below the configured minimum) or
  `TBD — requires real recording`.
