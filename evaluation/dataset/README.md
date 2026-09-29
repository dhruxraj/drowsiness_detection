# Evaluation dataset: recording protocol, label format and privacy

This document specifies how evaluation recordings are made, described and labelled so that
the evaluation in [`../README.md`](../README.md) is reproducible.

> **Status:** no recordings exist yet. `tuning/` and `final/` contain only `.gitkeep`, and
> the four CSV files in `evaluation/labels/` contain only their header rows. Nothing in this
> repository is real evaluation data until it has been recorded and labelled as described
> here.

## Contents

1. [Safety](#safety)
2. [Subjects and splits](#subjects-and-splits)
3. [Conditions](#conditions)
4. [Recording set-up](#recording-set-up)
5. [Recording script](#recording-script)
6. [Placing the files](#placing-the-files)
7. [Recording manifest format](#recording-manifest-format)
8. [Label format](#label-format)
9. [Labelling guide](#labelling-guide)
10. [Consent and privacy](#consent-and-privacy)
11. [Checking the dataset](#checking-the-dataset)

## Safety

Record **only at a desk or in a parked vehicle**. Never record or test while driving
(see the safety notice in the main README and CONTRIBUTING.md). Prolonged eye closure and
head drops in the script are acted.

## Subjects and splits

* Give every person a pseudonymous ID (`S01`, `S02`, ...). Never use names, initials or
  other identifying strings in IDs, file names or notes.
* **Assign each subject to exactly one split before recording** - `tuning` (used to choose
  thresholds) or `final` (held out, evaluated once after the thresholds are frozen). All
  recordings of a subject go into the same split. The tools reject a subject ID, recording
  ID, file or file content that appears in both splits.
* Do not move a subject between splits after any of their data has been looked at.
* More subjects give more reliable results. The report flags fewer than 10 subjects as a
  limitation; a practical starting point is roughly one third of the subjects for tuning
  and the rest for the final evaluation, with the full range of conditions in both splits.

## Conditions

Each recording has **one fixed set of conditions**. Variation is obtained by recording
several sessions per subject. Allowed values are defined in
`evaluation/config/evaluation_config.yaml -> vocabulary` (extend the lists there if you add
a condition - every value is reported separately):

| Attribute | Values | Meaning |
|---|---|---|
| `lighting` | `normal_indoor`, `dim_indoor`, `bright_frontal`, `backlit`, `side_lit`, `night_ir`, `daylight_vehicle` | dominant lighting on the face during the whole recording |
| `glasses` | `none`, `clear`, `sunglasses` | eyewear worn during the whole recording |
| `head_position` | `frontal`, `yaw_offset`, `pitch_offset`, `varied` | baseline head position relative to the camera (e.g. camera mounted off-axis gives `yaw_offset`) |
| `camera_distance` | `near` (< 50 cm), `medium` (50-70 cm), `far` (> 70 cm) | face-to-camera distance; also record the measured value in `camera_distance_cm` |

Suggested session matrix per subject (adapt to what the subject can provide):

| Session | lighting | glasses | head_position | camera_distance |
|---|---|---|---|---|
| 1 (baseline) | normal_indoor | none | frontal | medium |
| 2 | dim_indoor | none | frontal | medium |
| 3 | backlit or side_lit | none | frontal | medium |
| 4 | normal_indoor | clear (or sunglasses) | frontal | medium |
| 5 | normal_indoor | none | yaw_offset | medium |
| 6 | normal_indoor | none | frontal | near or far |

Head movement, blinking, eye closure, yawning and temporary face loss are covered **inside
every session** by the script below.

## Recording set-up

* Mount the camera as described in the main README, section 5 (in front of the subject,
  40-80 cm away, at or slightly below eye level), except where the session deliberately
  varies position or distance.
* Record with the camera and resolution you intend to evaluate; the project default is
  640×480 at 30 FPS (`config.yaml -> camera`). Record the camera model in the manifest.
* Save the **raw** video (no beauty filters, no stabilisation, no cropping). Prefer a
  constant frame rate. If your recorder writes variable-frame-rate files, convert them
  before labelling so that labels and processing use the same timeline, for example
  `ffmpeg -i in.mp4 -vf fps=30 -c:v libx264 -crf 18 -an out.mp4`.
* Do not cut the file after labelling; the SHA-256 hash in the manifest detects changes.
* Note anything unusual (e.g. a second person visible) in `notes` without personal details.

## Recording script

Timings are guidance; what counts are the labelled timestamps. Leave at least 10-15 s of
normal driving posture between events so the alarm can switch off (the alarm lasts at
least `alarm.min_duration` = 3 s and needs `alarm.release_time` of alertness to stop).
A session following this script lasts about 7-8 minutes.

| # | Instruction | Label (`event_type` / `alarm_expected`) |
|---|---|---|
| 1 | **Calibration** (start of file): sit normally, look straight ahead, eyes open, mouth closed for at least 8 s (the system calibrates for `calibration.duration` = 5 s). | none; set `calibration_end_s` (e.g. 8.0) - nothing before it is evaluated |
| 2 | Sit normally, look ahead, blink naturally for 60 s. | `blink` / `no` (one interval covering the period) |
| 3 | Three brief eye closures of about 1 s, ~10 s apart. | `eye_closure` / `no` (each ≤ 1.0 s) |
| 4 | Three prolonged eye closures of about 4 s, ~20 s apart. | `eye_closure` / `yes` (each ≥ 3.0 s) |
| 5 | Talk or read aloud for 20 s. | `talking` / `no` |
| 6 | Three wide yawns (real or acted), ~15 s apart. | `yawn` / `no` |
| 7 | Head movement: look at the left mirror for 3 s, right mirror for 3 s, glance down at the dashboard for < 1 s, look up for 2 s. | `head_movement` / `no` (one interval per movement) |
| 8 | Three brief nods (< 1 s). | `head_drop` / `no` |
| 9 | Two sustained head drops: let the head sink forward and hold for about 4 s. | `head_drop` / `yes` (each ≥ 3.0 s) |
| 10 | Temporary face loss: cover the camera (or lean out of view) for 2 s, then later for 8 s. | `face_loss` / `no` |
| 11 | Sit normally for 60 s. | none (unlabelled = normal) or `normal` / `no` |

Durations between the brief and prolonged thresholds (1.0-3.0 s closures, head drops
< 3.0 s that are more than a nod) are ambiguous and must be labelled `ignore`.

## Placing the files

```
evaluation/
├── dataset/
│   ├── tuning/                     # videos of TUNING subjects (git-ignored)
│   │   └── S01_baseline_01.mp4
│   └── final/                      # videos of FINAL subjects (git-ignored)
│       └── S07_dim_01.mp4
└── labels/
    ├── tuning_recordings.csv       # manifest of tuning/  (committed; no personal data)
    ├── tuning_labels.csv           # labels of tuning recordings
    ├── evaluation_recordings.csv   # manifest of final/
    └── evaluation_labels.csv       # labels of final recordings
```

Recommended `recording_id`: `<subject>_<session>_<nn>`, e.g. `S01_baseline_01`; use it as
the file name. The `file` column holds the path relative to the split folder
(sub-folders allowed, absolute paths and `..` are rejected).

## Recording manifest format

`evaluation/labels/tuning_recordings.csv` and `evaluation/labels/evaluation_recordings.csv`
- one row per recording, UTF-8, comma-separated, header row as in the template.

| Column | Required | Description |
|---|---|---|
| `recording_id` | yes | unique ID (letters, digits, `_ . -`), unique across both splits |
| `subject_id` | yes | pseudonymous subject ID (e.g. `S01`); a subject may appear in one split only |
| `split` | yes | `tuning` in the tuning manifest, `final` in the evaluation manifest |
| `file` | yes | video path relative to `evaluation/dataset/<split>/` |
| `sha256` | recommended | file hash; filled in by `validate_dataset.py --write-hashes` |
| `duration_s` | yes | recording duration in seconds |
| `source_fps` | yes | frame rate of the video file |
| `resolution` | yes | `WIDTHxHEIGHT`, e.g. `640x480` |
| `camera_model` | recommended | camera make/model or `laptop built-in` |
| `lighting`, `glasses`, `head_position`, `camera_distance` | yes | conditions (see [Conditions](#conditions)) |
| `camera_distance_cm` | optional | measured face-camera distance |
| `calibration_end_s` | yes | end of the start-up calibration period; evaluation starts here |
| `consent` | yes | must be `yes` (documented informed consent exists) |
| `recorded_on` | optional | date `YYYY-MM-DD` |
| `annotator` | optional | pseudonymous annotator ID |
| `notes` | optional | free text without personal information |

Example (illustrative only - this recording does not exist):

```csv
recording_id,subject_id,split,file,sha256,duration_s,source_fps,resolution,camera_model,lighting,glasses,head_position,camera_distance,camera_distance_cm,calibration_end_s,consent,recorded_on,annotator,notes
S01_baseline_01,S01,tuning,S01_baseline_01.mp4,,452.3,30,640x480,Logitech C920,normal_indoor,none,frontal,medium,60,8.0,yes,2026-10-01,A1,
```

## Label format

`evaluation/labels/tuning_labels.csv` and `evaluation/labels/evaluation_labels.csv` - one
row per labelled interval.

| Column | Required | Description |
|---|---|---|
| `recording_id` | yes | must exist in the manifest of the same split |
| `event_id` | yes | unique within the recording (e.g. `e01`) |
| `event_type` | yes | see table below |
| `start_s` | yes | start in seconds on the video clock (0 = first frame) |
| `end_s` | yes | end in seconds; must be > `start_s` and within the recording |
| `alarm_expected` | yes | `yes`, `no` or `ignore` (allowed values per type below) |
| `annotator` | optional | pseudonymous annotator ID |
| `notes` | optional | free text without personal information |

| `event_type` | Represents | Allowed `alarm_expected` |
|---|---|---|
| `normal` | explicitly labelled non-event period (optional - unlabelled time is also normal) | `no` |
| `blink` | a period of normal, natural blinking | `no` |
| `eye_closure` | eyes closed | `yes` = prolonged (≥ 3.0 s), `no` = brief (≤ 1.0 s), `ignore` = in between |
| `yawn` | yawning | `no`, `ignore` |
| `talking` | talking / reading aloud | `no`, `ignore` |
| `head_movement` | looking sideways/up/down, glances | `no`, `ignore` |
| `head_drop` | head sinking forward | `yes` = sustained (≥ 3.0 s), `no` = brief nod, `ignore` |
| `face_loss` | face deliberately covered / out of view | `no`, `ignore` |
| `other` | anything else worth marking | `no`, `ignore` |

The durations come from `evaluation_config.yaml -> ground_truth`. They are protocol
definitions fixed before tuning and do not depend on the detector's thresholds; the loader
warns (does not fail) when a label contradicts them.

Rules enforced by the validator: alarm-expected events must not overlap each other and
must not start before `calibration_end_s`; other labels may overlap (e.g. a `head_movement`
during a `blink` period).

Example (illustrative only):

```csv
recording_id,event_id,event_type,start_s,end_s,alarm_expected,annotator,notes
S01_baseline_01,e01,blink,10.0,70.0,no,A1,
S01_baseline_01,e02,eye_closure,82.4,83.3,no,A1,brief closure
S01_baseline_01,e03,eye_closure,121.6,125.9,yes,A1,prolonged closure
S01_baseline_01,e04,yawn,190.2,194.0,no,A1,
S01_baseline_01,e05,face_loss,402.0,410.1,no,A1,camera covered
```

## Labelling guide

* Use a video player or annotation tool that can step frame by frame and shows the
  timestamp of the current frame (time since the first frame). Record times in seconds
  with at least two decimals.
* **eye_closure:** start = first frame in which the eyelids are (almost) fully closed;
  end = first frame in which the eyes are clearly open again.
* **blink:** label the *period* of natural blinking (e.g. the 60 s of step 2), not each
  blink.
* **yawn:** start = mouth starts to open wide; end = mouth closed again.
* **head_drop:** start = head starts to sink below the neutral position; end = head back
  in the neutral position.
* **head_movement:** start = the head starts turning; end = back to the neutral position.
* **face_loss:** start = first frame in which the face is no longer fully visible; end =
  first frame in which it is fully visible again.
* Decide `alarm_expected` from the **measured** duration using the table above; use
  `ignore` for ambiguous cases rather than guessing.
* Do not look at the system's output (dashboard, logs, traces) while labelling.
* Ideally a second person labels a sample of recordings independently; disagreements
  larger than a few frames should be discussed and the protocol clarified.

## Consent and privacy

Recordings show identifiable faces and are personal (biometric) data.

Before recording, obtain written informed consent that covers at least:

* what is recorded (face video) and why (evaluating this prototype),
* where the data is stored, who can access it and for how long,
* that raw recordings will not be published and only aggregate results are published,
* whether per-subject results (under a pseudonymous ID) may be published,
* that consent can be withdrawn and the recordings deleted.

Handling rules:

* Record `consent = yes` in the manifest only if such consent is documented; the validator
  rejects anything else.
* **Never commit recordings.** `.gitignore` excludes everything in `evaluation/dataset/tuning/`
  and `evaluation/dataset/final/` (except `.gitkeep`), video files anywhere under
  `evaluation/`, and `evaluation/results/runs/` (per-frame traces derived from the
  recordings). Check `git status` before every commit.
* Keep the mapping between pseudonymous IDs and real people outside the repository.
* Store the raw videos on encrypted storage and delete them at the end of the agreed
  retention period or when consent is withdrawn (then also remove the rows from the
  manifest/labels and re-run the evaluation).
* Share recordings with other contributors only if the consent allows it.

## Checking the dataset

```bash
python evaluation/scripts/validate_dataset.py                 # schema, vocabulary, consent, separation
python evaluation/scripts/validate_dataset.py --check-files   # + every video file exists
python evaluation/scripts/validate_dataset.py --write-hashes  # fill in / verify sha256
python evaluation/scripts/validate_dataset.py --probe         # compare fps, resolution, duration with the files
```

The validator lists every problem at once and exits with code 2 if anything is invalid or
if the tuning and final splits overlap.
