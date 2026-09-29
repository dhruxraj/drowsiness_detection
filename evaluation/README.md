# Multi-condition system evaluation

This folder contains the framework for evaluating the **complete** drowsiness detection
system on labelled recordings from multiple subjects and environmental conditions
(issue #4). It complements the automated tests in `tests/`, which verify that the
detection logic behaves as designed but do not establish real-world performance.

> **Current status: no real labelled recordings have been collected yet.** The framework,
> label format, recording protocol and metric code are complete and tested with synthetic
> unit-test fixtures only. Every result in `evaluation/results/` is therefore
> **"Not yet measured — real labelled recordings required"**. No experimental result has
> been estimated or filled in. Section [Reproducing the evaluation](#reproducing-the-evaluation)
> describes how to produce the real numbers.

## Contents

1. [What is evaluated](#what-is-evaluated)
2. [Folder structure](#folder-structure)
3. [Workflow overview](#workflow-overview)
4. [Dataset and labels](#dataset-and-labels)
5. [Separation of tuning and final data](#separation-of-tuning-and-final-data)
6. [Thresholds: selection and freezing](#thresholds-selection-and-freezing)
7. [Event matching](#event-matching)
8. [Metric definitions](#metric-definitions)
9. [Breakdown by condition](#breakdown-by-condition)
10. [Reproducing the evaluation](#reproducing-the-evaluation)
11. [Outputs](#outputs)
12. [Privacy and data handling](#privacy-and-data-handling)
13. [Limitations](#limitations)
14. [Tests](#tests)
15. [Maintenance](#maintenance)

---

## What is evaluated

The evaluator runs every recording through the **production pipeline**, frame by frame,
using the modules in `src/` in the same order as the per-frame loop in `main.py`:

```
src.camera.Camera (video file, video clock - same as `python main.py --source <file>`)
  -> Preprocessor.process -> FaceLandmarkDetector.detect -> HeadPoseEstimator.estimate
  -> compute_ear / compute_mar -> FrameMetrics
  -> Calibrator (start-up calibration) | TemporalAnalyzer.update -> DrowsinessScorer.compute
     -> DecisionEngine.update  ->  alarm_on
```

There is no separate detector for evaluation. `evaluation/core/pipeline.py` only composes
the production classes; detection thresholds come from `config.yaml` and are loaded with
the production loader (`src.config.load_config`), so the same validation applies.

Deliberately **not** part of an evaluation run: the dashboard window, the physical alarm
backends and the session logger. The alarm is taken at the moment the decision engine
switches `alarm_on` from off to on, i.e. the software alarm decision. Hardware delays of
speakers, GPIO or serial devices are therefore not included (see [Limitations](#limitations)).

`tests/test_evaluation_pipeline_sync.py` fails if the mirrored calls disappear from
`main.py` or change order, so a change to the per-frame loop cannot silently diverge
from the evaluator.

## Folder structure

```
evaluation/
├── README.md                    # this file: methodology and reproduction
├── config/
│   ├── evaluation_config.yaml   # protocol: paths, label vocabulary, matching tolerances, tuning grid
│   └── frozen_thresholds.yaml   # snapshot + fingerprints of the thresholds frozen before final evaluation
├── dataset/
│   ├── README.md                # recording protocol, label schema, labelling guide, consent/privacy
│   ├── tuning/                  # tuning recordings (git-ignored, never committed)
│   └── final/                   # held-out final recordings (git-ignored, never committed)
├── labels/
│   ├── tuning_recordings.csv    # one row per tuning recording (subject, conditions, camera, fps ...)
│   ├── tuning_labels.csv        # labelled event intervals of the tuning recordings
│   ├── evaluation_recordings.csv# one row per final recording
│   └── evaluation_labels.csv    # labelled event intervals of the final recordings
├── scripts/
│   ├── validate_dataset.py      # schema, vocabulary, consent, file hashes, split separation
│   ├── evaluate.py              # run recordings through the production pipeline + metrics
│   ├── calculate_metrics.py     # recompute metrics/reports from a saved run (no MediaPipe needed)
│   ├── tune.py                  # grid search on the TUNING split (trace replay)
│   └── freeze_thresholds.py     # freeze config.yaml thresholds before the final evaluation
├── core/                        # library used by the scripts (see core/__init__.py)
└── results/
    ├── README.md                # what the result files contain
    ├── metrics.csv              # machine-readable overall metrics (official final results only)
    ├── breakdown.csv            # machine-readable metrics per condition
    ├── report.md                # human-readable report
    └── runs/                    # every run with per-frame traces (git-ignored)
```

## Workflow overview

```
record + label  ->  validate  ->  evaluate TUNING  ->  tune (TUNING only)  ->  update config.yaml
     ->  re-evaluate TUNING  ->  freeze thresholds + commit  ->  evaluate FINAL once  ->  publish
```

1. Record subjects following the script in [`dataset/README.md`](dataset/README.md), assign
   each **subject** to either the tuning or the final split before recording, and label the
   recordings.
2. `validate_dataset.py` checks both splits, fills in SHA-256 hashes and rejects overlap.
3. `evaluate.py --split tuning` runs the tuning recordings; `tune.py` searches the
   configured threshold grid on those traces only.
4. Apply the chosen values to `config.yaml` by hand, re-run the tuning evaluation to confirm.
5. `freeze_thresholds.py` records the thresholds (and the evaluation protocol) with
   fingerprints; commit `config.yaml` and `frozen_thresholds.yaml` together.
6. `evaluate.py --split final` refuses to run unless the thresholds are frozen and unchanged.
   A complete final run is an *official* result and is published to `evaluation/results/`.

## Dataset and labels

The complete specification is in [`dataset/README.md`](dataset/README.md). In short:

* **Recording manifest** (`*_recordings.csv`), one row per recording:
  `recording_id, subject_id, split, file, sha256, duration_s, source_fps, resolution,
  camera_model, lighting, glasses, head_position, camera_distance, camera_distance_cm,
  calibration_end_s, consent, recorded_on, annotator, notes`.
  Each recording has one fixed set of conditions (lighting, glasses, head position,
  camera distance); variation comes from recording several sessions.
* **Labels** (`*_labels.csv`), one row per labelled interval:
  `recording_id, event_id, event_type, start_s, end_s, alarm_expected, annotator, notes`.
  Times are seconds on the video clock (0 = first frame). **Unlabelled time is treated as
  normal, non-event time.**
* **Event types:** `normal`, `blink` (period of natural blinking), `eye_closure`, `yawn`,
  `talking`, `head_movement`, `head_drop`, `face_loss`, `other`.
* **`alarm_expected`**: `yes` (the alarm must be raised - prolonged eye closure or sustained
  head drop), `no` (the alarm must not be raised), `ignore` (ambiguous; excluded from
  false-alarm and missed-event counting). Allowed values per event type are defined in
  `evaluation_config.yaml -> vocabulary.event_types`; for example a yawn can never be `yes`
  because the system is designed not to alarm on yawning alone.
* The protocol durations that decide `yes` / `no` / `ignore` (`ground_truth` in
  `evaluation_config.yaml`: prolonged closure ≥ 3.0 s, brief closure ≤ 1.0 s, sustained
  head drop ≥ 3.0 s) are fixed before tuning and are independent of the detector thresholds.
  The loader warns when a label contradicts them.

Validation rejects (and lists all at once): missing columns, IDs that look like names,
unknown vocabulary values, missing consent, paths outside the split folder, events outside
the recording, alarm-expected events inside the calibration period, overlapping
alarm-expected events, duplicate IDs and non-numeric values.

## Separation of tuning and final data

The issue requires that subjects and recordings used for tuning are separate from those
used for the final evaluation. This is enforced in code (`evaluation/core/splits.py`):

* Each split has its own manifest, label file and dataset folder; every manifest row
  carries its `split`, and a row in the wrong manifest is rejected.
* `validate_dataset.py`, `evaluate.py`, `calculate_metrics.py`, `tune.py` and
  `freeze_thresholds.py` all refuse to run when the two splits share a **recording ID**,
  a **subject ID** (case-insensitive - subject-level separation, not only recording-level),
  the **same file path**, or **identical file content** (SHA-256).
* `tune.py` only accepts runs of the tuning split and refuses runs that contain final
  recordings.
* The frozen-threshold record lists the tuning recordings and subjects; the final
  evaluation is rejected if any of them appears in the final split.
* Tuning-split results are never published as final results.

Which recordings were used for tuning and which for the final evaluation is documented in
`frozen_thresholds.yaml` (`tuning_recordings`, `tuning_subjects`), in each run's
`run.json`, and in the "Recordings used" table of `report.md`.

## Thresholds: selection and freezing

### Where thresholds live

All detection thresholds stay in the project's single configuration file, `config.yaml`
(as required by CONTRIBUTING.md). The evaluation does not introduce its own thresholds.
`report.md` lists the key thresholds with their current values and why each exists:

| Parameter | Why it exists |
|---|---|
| `eyes.ear_threshold` | Fallback closed-eye EAR limit when calibration is off or fails. |
| `calibration.ear_ratio`, `ear_min`, `ear_max` | Personal closed-eye threshold = ratio × the driver's open-eye EAR, clamped (eye shape differs between people). |
| `eyes.ear_hysteresis` | Eyes count as re-opened only above threshold + hysteresis (prevents flicker). |
| `eyes.blink_max_duration` | Closures shorter than this are normal blinks and add 0 to the eye score. |
| `eyes.long_blink_min_duration` | Closures at least this long count as long blinks (fatigue history). |
| `eyes.closure_duration_threshold` | Continuous closure that on its own raises the alarm (micro-sleep). |
| `eyes.max_yaw_for_eye_analysis` | Beyond this head turn EAR is unreliable and eye evidence is ignored. |
| `mouth.mar_threshold`, `calibration.mar_margin` | Mouth-wide-open limit (personal: max(threshold, resting MAR + margin)). |
| `mouth.yawn_min_duration` | Mouth must stay open this long to count as a yawn (talking is shorter). |
| `head.pitch_down_threshold` | Head pitched down more than this vs. the calibrated neutral = head drop. |
| `head.nod_min_duration` | Head dips shorter than this are ignored. |
| `head.head_drop_duration_threshold` | Sustained head drop that on its own raises the alarm. |
| `head.yaw_distraction_threshold` | Beyond this yaw the state is LOOKING AWAY (never drowsiness). |
| `face_loss.grace_period`, `face_loss.warn_after` | Short dropouts are ignored; long face loss gives a warning, never a drowsiness alarm. |
| `scoring.threshold`, `release_threshold`, `cumulative_cap` | Alarm-on score, release condition, cap so fatigue history alone never alarms. |
| `alarm.min_duration`, `alarm.release_time` | Minimum alarm duration and required alert time before it stops. |

The current values are the design defaults documented in the main README; they have **not**
been tuned on real labelled recordings yet.

### Selecting thresholds (tuning split only)

`tune.py` replays the per-frame measurements recorded in a tuning run (EAR, MAR, head pose,
face presence) through the production calibration / temporal analysis / scoring / decision
code for every combination in `evaluation_config.yaml -> tuning.search_space`. Candidates
are ranked by:

1. feasibility: false alarms per hour ≤ `tuning.max_false_alarms_per_hour` (a design target
   chosen before tuning; review it before use),
2. fewest missed events, then fewest false alarms per hour, then lowest median latency.

Before searching, the unchanged configuration is replayed and must reproduce the recorded
alarm sequence exactly, otherwise the script stops (replay would not be faithful). Only
parameters that act *after* the per-frame measurement can be tuned this way; `camera.*`,
`preprocessing.*`, `face_mesh.*`, `head.invert_pitch` and `head.enabled` are rejected
because changing them changes the measurements - evaluate such changes with a new tuning
run instead. `tune.py` never edits `config.yaml`; it writes `tuning_results.csv` and
`best_overrides.yaml` into the run directory.

### Freezing

`freeze_thresholds.py` writes `evaluation/config/frozen_thresholds.yaml` with:

* the complete detection-relevant sections of `config.yaml` (`preprocessing`, `face_mesh`,
  `eyes`, `mouth`, `head`, `face_loss`, `calibration`, `scoring`, `alarm.min_duration`,
  `alarm.release_time`) and their SHA-256 fingerprint,
* the evaluation protocol (`matching`, `ground_truth`) and its fingerprint,
* the tuning recordings and subjects, the tuning manifest hash, the selection method, the
  tuning run, the git commit and time of freezing.

`evaluate.py --split final` refuses to run (exit code 3) unless the file has
`status: frozen`, both fingerprints still match, and no final recording or subject was used
for tuning. Re-freezing an already frozen file requires `--force`, and the record notes
whether final results already existed at the time of freezing. If the defaults are frozen
without tuning, `--untuned-defaults` must be given and the record says so.

## Event matching

The system produces **temporal alarms** (the alarm switches on, stays on for at least
`alarm.min_duration`, then switches off with hysteresis). Matching is therefore done on
alarm **episodes**, not on individual frames (`evaluation/core/matching.py`).

* An **alarm episode** starts at the first frame with `alarm_on = true` after a frame with
  `alarm_on = false` (the *onset*) and ends at the next frame with `alarm_on = false`.
* Every alarm-expected event `[start, end]` has a **matching window**
  `[start − pre_tolerance_s, end + post_tolerance_s]` (inclusive).
* Onsets are processed in time order:
  * an onset inside the window of an event that has not been detected yet **detects** that
    event (if several windows overlap, the earliest-starting undetected event is chosen;
    one onset detects at most one event);
  * an onset inside the window of an event that is already detected is a **duplicate**
    (reported separately, not a false alarm);
  * otherwise, an onset inside the window of an `ignore` event (same tolerances) is
    **ignored**;
  * any other onset inside the evaluation window is a **false alarm**;
  * onsets before `calibration_end_s` are counted separately as alarms during calibration.
* An alarm-expected event without a detecting onset is a **missed event**.
* The rule is onset-based and conservative: an alarm that was already on before an event's
  window opened (e.g. still sounding from an earlier false alarm) does not detect the
  event.

Default tolerances (configurable in `evaluation_config.yaml -> matching`):

| Setting | Default | Reason |
|---|---|---|
| `pre_tolerance_s` | 0.5 s | An annotated start can be a few frames late (annotator reaction, frame stepping); an onset shortly before the labelled start is still a response to that event. |
| `post_tolerance_s` | 1.0 s | An annotated end can be a few frames early; alarm decisions may occur in the last frames before the eyes re-open or the head comes up. |

The tolerances are part of the frozen protocol; changing them after freezing invalidates
the final evaluation.

## Metric definitions

All metrics are computed in `evaluation/core/metrics.py` from per-frame traces plus labels.
Units: seconds (`_s`), milliseconds (`_ms`), percent (`_pct`), frames per second (FPS).

**Evaluation window.** For each recording: from `calibration_end_s` (end of the labelled
start-up calibration, during which the system cannot alarm by design) to the end of the
last processed frame. Recordings that could not be processed are excluded from the alarm
metrics and listed separately (see *complete recording failure* below).

### 1. False alarms per hour

```
false_alarms_per_hour = false_alarms / non_event_hours
non_event_time = evaluation window
                 − union of matching windows of alarm-expected events
                 − union of matching windows of ignore events
```

A false alarm is an alarm onset classified as such by the matching rule above, i.e. an
onset during time where the ground truth contains no corresponding alarm-expected event.
Normalising by *non-event* time (not total time) measures the rate during periods in which
an alarm is not wanted. `false_alarms_per_evaluated_hour` (normalised by the full
evaluation window minus ignore windows) is reported as well. False alarms are also broken
down by the labelled context at the onset (e.g. `yawn`, `head_movement`, `face_loss`,
unlabelled).

### 2. Missed events

The number of alarm-expected events (`alarm_expected = yes`) without a detecting onset
under the matching rule, together with the detection rate
(`detected / alarm-expected events`), per event type, and how many missed events had the
face/measurements unavailable for more than half of the event.

### 3. Alarm trigger latency

For every detected event: `latency_ms = (alarm_onset_s − event_start_s) × 1000`.
Reported as mean, median, minimum, maximum (and 90th percentile), with the number of
detected events. Latency can be negative when the onset falls within the pre-tolerance.
By design, eye closure alone raises the alarm once the continuous closure reaches
`eyes.closure_duration_threshold`, so latencies of that order are expected; the evaluation
measures the actual values.

As a supplementary figure, the **alarm recovery time** (listed in the main README's planned
metrics) is reported for detected events as `alarm_offset − event_end` in milliseconds
(`release_delay_ms`): how long the alarm keeps sounding after the labelled event ended,
which reflects `alarm.min_duration` and `alarm.release_time`.

### 4. Measurement / face-loss rate

Frame-based, over the frames inside the evaluation window:

| Category | Definition |
|---|---|
| face missing | the landmark detector returned no face for the frame |
| invalid measurement | a face was found but EAR or MAR is not a finite number, or head pose is unavailable while `head.enabled` is true (reasons are counted) |
| measurement unavailable | face missing + invalid measurement |
| unexpected unavailable | unavailable frames **outside** labelled `face_loss` intervals |

The same quantities are also given as a share of time. Continuous face-missing runs are
classified with the thresholds of `config.yaml -> face_loss`:

* **short dropout**: shorter than `face_loss.grace_period` (ignored by the analyzer),
* **temporary face loss**: shorter than `face_loss.warn_after`,
* **extended face loss**: `face_loss.warn_after` or longer (DRIVER NOT VISIBLE warning),

and marked *expected* if they overlap a labelled `face_loss` interval.

**Complete recording failure** is reported per recording, never mixed into the frame rates:

| Status | Meaning | In alarm metrics? |
|---|---|---|
| `ok` | processed | yes |
| `no_face_detected` | processed, but the face was never found | yes - its alarm-expected events count as missed |
| `failed_to_open` | camera/detector could not be created for the file | no - listed as failed |
| `no_frames` | file opened, no frame decoded | no - listed as failed |
| `stalled` | frames stopped arriving without end-of-file (`runner.stall_timeout_s`) | no - listed as failed |
| `failed_during_processing` | an exception in the pipeline | no - listed as failed |

### 5. Processing frame rate

For every frame, `processing_ms` is the wall-clock time (`time.perf_counter`) of the
pipeline work: pre-processing, landmarks, EAR/MAR/head pose, calibration or temporal
analysis, scoring and decision. Frame acquisition/decoding is measured separately
(`acquire_ms`). Dashboard, alarm hardware and logging are not part of an evaluation run.

| Metric | Definition |
|---|---|
| `processing_fps_mean` | total frames / total processing time (pooled over all evaluated recordings) |
| `processing_fps_median` | 1000 / median `processing_ms` |
| `processing_fps_p5` | 1000 / 95th percentile of `processing_ms` (a "slow frame" rate) |
| `processing_fps_min_recording` | lowest per-recording mean processing FPS |
| `end_to_end_fps` | total frames / (processing + acquisition time) |
| `source_fps_median` | frame rate of the recordings (manifest), **not** a performance figure |

Processing FPS is what the pipeline achieved on the machine that ran the evaluation (Python,
platform, CPU count and package versions are recorded in `run.json` and the report). It is
not a theoretical value, and it is distinct from the source video frame rate: offline
processing of a file is not limited to real time.

### Supplementary: analyzer indicator events

Not an alarm metric: for all labelled `yawn`, `eye_closure` and `head_drop` events (any
`alarm_expected` value), whether the
temporal analyzer logged the corresponding event message (`YAWN STARTED`, `MICROSLEEP` /
`LONG BLINK`, `HEAD DROP`) within the matching window.

## Breakdown by condition

All metrics are also computed per value of `subject_id`, `lighting`, `glasses`,
`head_position` and `camera_distance` (`breakdown.csv`, report section 5). Groups with
fewer than `breakdowns.min_positive_events` alarm-expected events or less than
`breakdowns.min_non_event_minutes` of non-event time are flagged as *insufficient data*.
Conditions from the vocabulary that are not present in the dataset are listed as
limitations in the report.

## Reproducing the evaluation

All commands run from the repository root with the project's virtual environment active.

```bash
# 0. install (same as the main README) and check the test suite
pip install -r requirements.txt
python -m unittest discover -s tests -t . -v

# 1. place recordings and fill in the manifests/labels (see evaluation/dataset/README.md)
#    evaluation/dataset/tuning/<file>, evaluation/dataset/final/<file>
#    evaluation/labels/{tuning,evaluation}_{recordings,labels}.csv

# 2. validate: schema, vocabulary, consent, files, split separation; fill in SHA-256 hashes;
#    compare fps / resolution / duration with the video files
python evaluation/scripts/validate_dataset.py --check-files
python evaluation/scripts/validate_dataset.py --write-hashes
python evaluation/scripts/validate_dataset.py --probe

# 3. evaluate the TUNING split (results stay in evaluation/results/runs/<run_id>/)
python evaluation/scripts/evaluate.py --split tuning

# 4. search thresholds on the tuning traces only
python evaluation/scripts/tune.py --run-dir evaluation/results/runs/<run_id>

# 5. apply the chosen values to config.yaml by hand, confirm on the tuning split
python evaluation/scripts/evaluate.py --split tuning

# 6. freeze, then commit config.yaml + evaluation/config/frozen_thresholds.yaml
python evaluation/scripts/freeze_thresholds.py \
    --selection-method "grid search (tune.py), see <run_id>/tuning/tuning_results.csv" \
    --tuning-run evaluation/results/runs/<run_id>
#   (or, to evaluate the untuned defaults, say so explicitly:)
#   python evaluation/scripts/freeze_thresholds.py --untuned-defaults \
#       --selection-method "config.yaml design defaults, not tuned on data"

# 7. evaluate the FINAL split once; an official result is published to evaluation/results/
python evaluation/scripts/evaluate.py --split final \
    --dataset evaluation/dataset/final \
    --recordings evaluation/labels/evaluation_recordings.csv \
    --labels evaluation/labels/evaluation_labels.csv \
    --config evaluation/config/evaluation_config.yaml

# 8. recompute metrics/reports from a saved run (e.g. after correcting a label error),
#    optionally re-publish; no MediaPipe needed
python evaluation/scripts/calculate_metrics.py --run-dir evaluation/results/runs/<run_id> --publish

# regenerate the "not yet measured" result files
python evaluation/scripts/calculate_metrics.py --placeholder
```

`evaluate.py` options: `--only <recording_id ...>` processes a subset (debugging; never
official), `--frame-step N` processes only every N-th frame of the same recordings
(timestamps stay on the video clock) to check that the time-based temporal analysis gives
the same alarms at a lower processing frame rate, as suggested in the main README's planned
evaluation (never official; compare its `metrics.csv` with the full-rate run),
`--out <dir>` sets the run directory, `--system-config <file>` evaluates another
detection config (tuning split), `--allow-unfrozen` runs the final split without frozen
thresholds (marked NOT official and never published), `--no-publish`.

Exit codes: `0` success, `1` nothing to evaluate / missing dependency, `2` invalid dataset or
tuning/final leakage, `3` thresholds not frozen or changed since freezing / not publishable.

## Outputs

Every run writes a new directory `evaluation/results/runs/<UTC timestamp>_<split>[_step<N>]/`
(an existing run is never overwritten; the directory is git-ignored because the traces are
derived from biometric recordings):

| File | Content |
|---|---|
| `run.json` | split, git commit and dirty flag, machine and package versions, config fingerprints, landmark engine, per-recording status/errors/frame counts |
| `traces/<recording_id>.csv` | one row per frame: timestamp, face detected, EAR, MAR, pitch/yaw/roll, measurement validity and reason, calibrating, state, alarm_on, score, analyzer events, `processing_ms`, `acquire_ms` |
| `metrics.csv` | overall metrics: `scope, group, metric, value, unit, sample_size, status` |
| `breakdown.csv` | the same columns per condition value |
| `report.md` | human-readable report (dataset, protocol, thresholds, results, breakdowns, environment, limitations) |
| `summary.json` | all computed quantities |
| `matches.csv` | every alarm-expected event: detected / missed / not evaluated, onset, latency |
| `alarms.csv` | every alarm episode: detection, false alarm (with context), duplicate, ignored, during calibration |
| `tuning/` | `tuning_results.csv`, `best_overrides.yaml` (tune.py) |

Only an **official** result - final split, all final recordings, frozen and unchanged
thresholds - is copied to `evaluation/results/` (`metrics.csv`, `breakdown.csv`,
`report.md`, `summary.json`). These contain aggregate numbers only and may be committed.
In `metrics.csv`, `status` is `measured`, `undefined` (e.g. latency with no detected
events), `insufficient_data` (breakdown groups) or `TBD — requires real recording`.

## Privacy and data handling

Recordings show identifiable faces and are personal (biometric) data.

* Recordings are **never committed**: `.gitignore` excludes everything in
  `evaluation/dataset/tuning/` and `evaluation/dataset/final/` (except `.gitkeep`), video
  files anywhere under `evaluation/`, and the run directories with per-frame traces.
* Only recordings with documented informed consent may be used (`consent = yes` is
  required by the validator). Use pseudonymous IDs (`S01`, `S02` ...) - the validator
  rejects IDs containing spaces - and keep the mapping to real identities outside the
  repository.
* Manifests and labels contain pseudonymous IDs, conditions and timestamps only. Keep
  free-text notes free of personal information.
* Published results are aggregates. Per-subject breakdowns use pseudonymous IDs only; drop
  `subject_id` from `breakdowns.attributes` if even that is not covered by the consent.

Details, including a consent checklist, are in [`dataset/README.md`](dataset/README.md).

## Limitations

Limitations that follow from the protocol and framework design (the report lists them
together with the dataset-specific ones it detects, such as few subjects, uncovered
conditions, a single camera model, short total duration or missing event types):

* Drowsiness behaviours are **acted** following the recording script. Posed eye closures,
  yawns and head drops can differ from genuine fatigue, so the results describe detection
  of the scripted behaviours, not validated drowsiness detection.
* Recordings are made at a desk or in a parked vehicle (testing while driving is not
  allowed), so vehicle vibration, changing daylight and real driving demands are not
  represented.
* Ground truth is **manually annotated**; boundaries are limited by frame rate and
  annotator judgement. Inter-annotator agreement is not computed by the framework.
* `alarm_expected` encodes the system's intended behaviour (alarm for prolonged closure or
  sustained head drop). Yawning alone is by design not expected to raise the alarm, so the
  evaluation does not judge whether yawning-only fatigue should be alarmed.
* Latency is measured to the software alarm decision, not to physical buzzer/speaker output
  (see issue #3 for alarm health checks).
* Processing FPS is measured offline on video files without the dashboard and live capture,
  on the evaluating machine. It must be re-measured on target hardware (e.g. Raspberry Pi).
* Offline evaluation reads every frame of the file; live operation always processes the
  newest camera frame, so frame dropping under load is not reproduced by the offline run.
* The adapter mirrors `main.py`; the drift test guards the mirrored calls, but a change to
  the per-frame loop must still be applied to `evaluation/core/pipeline.py`.
* Tuning by trace replay covers only parameters applied after the per-frame measurement.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

`tests/test_evaluation_*.py` cover label loading and invalid labels, event matching, false
alarms per hour, missed events, latency, face/measurement loss, processing FPS, failure
statuses, split separation, threshold freezing, report/placeholder output, the command-line
workflow on a synthetic run and the `main.py` drift guard. They use **synthetic unit-test
fixtures** (`tests/evaluation_fixtures.py`) that only check the metric code; they are not
evaluation data and produce no experimental results. They need neither a camera nor
MediaPipe.

## Maintenance

* **`main.py` per-frame loop changed:** mirror the change in `evaluation/core/pipeline.py`
  and update `MIRRORED_MAIN_CALLS`; `tests/test_evaluation_pipeline_sync.py` fails until
  then. A future refactor could move the per-frame processing into one shared `src` function
  used by both `main.py` and the evaluator.
* **New recording condition:** add the value to `vocabulary` in `evaluation_config.yaml`.
* **New event type:** add it to `vocabulary.event_types` with its allowed `alarm_expected`
  values (keep the values quoted: unquoted `yes`/`no` are YAML booleans).
* **New detection parameter in `config.yaml`:** if it lives in a new section, add the
  section to `DETECTION_SECTIONS` in `evaluation/core/thresholds.py` so it is frozen.
