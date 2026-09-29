# Multi-condition evaluation report

> **Status: Not yet measured — real labelled recordings required.** This file documents the evaluation set-up. It contains no experimental results.

| Item | Value |
|---|---|
| Generated (UTC) | 2026-09-29T19:03:45+00:00 |
| Split | final |
| Run ID | — |
| Git commit | — |
| Landmark engine | — |
| Frames processed | every frame |
| Detection config fingerprint | 5e8784fc806a0753 |

## 1. Dataset

No labelled recordings are present in this split yet. Not yet measured — real labelled recordings required.

| Quantity | Value |
|---|---|
| Subjects | 0 |
| Recordings | 0 |
| Total duration | 0.0 min |
| Camera models | — |
| Resolutions | — |
| Source FPS | — |

**lighting**

| lighting | recordings | subjects | minutes |
|---|---|---|---|
| — | 0 | 0 | 0.0 |

**glasses**

| glasses | recordings | subjects | minutes |
|---|---|---|---|
| — | 0 | 0 | 0.0 |

**head_position**

| head_position | recordings | subjects | minutes |
|---|---|---|---|
| — | 0 | 0 | 0.0 |

**camera_distance**

| camera_distance | recordings | subjects | minutes |
|---|---|---|---|
| — | 0 | 0 | 0.0 |

**Labelled events**

| event_type | count | total s | alarm_expected=yes | no | ignore |
|---|---|---|---|---|---|
| — | 0 | 0.0 | 0 | 0 | 0 |

## 2. Protocol

- Tuning and final recordings are kept in separate manifests; the evaluator rejects any shared recording ID, subject ID, file path or file hash.
- Event matching: an alarm onset detects an alarm-expected event if it lies in [start − 0.5 s, end + 1.0 s]; one onset per event.
- The labelled calibration period at the start of each recording is excluded from all alarm and face-loss metrics.
- Full definitions: `evaluation/README.md`.

**Recordings used**

| Split | Subjects | Recordings |
|---|---|---|
| tuning (threshold selection, from the frozen record) | — | — |
| final (this report) | — | — |

## 3. Thresholds

**Thresholds are not frozen yet.** The values below are the current `config.yaml` defaults; they have not been tuned on real labelled recordings.

| Parameter | Value | Why it exists |
|---|---|---|
| `eyes.ear_threshold` | 0.21 | Fallback closed-eye EAR limit when calibration is off or fails. |
| `calibration.ear_ratio` | 0.75 | Personal closed-eye threshold = ratio x the driver's open-eye EAR (eye shape differs between people). |
| `calibration.ear_min` | 0.15 | Lower clamp of the personal EAR threshold. |
| `calibration.ear_max` | 0.28 | Upper clamp of the personal EAR threshold. |
| `eyes.ear_hysteresis` | 0.02 | Eyes count as re-opened only above threshold + hysteresis (prevents flicker). |
| `eyes.blink_max_duration` | 0.4 | Closures shorter than this are normal blinks and add 0 to the eye score. |
| `eyes.long_blink_min_duration` | 0.5 | Closures at least this long count as long blinks (fatigue history). |
| `eyes.closure_duration_threshold` | 1.8 | Continuous closure that on its own raises the alarm (micro-sleep). |
| `eyes.max_yaw_for_eye_analysis` | 30 | Beyond this head turn EAR is unreliable and eye evidence is ignored. |
| `mouth.mar_threshold` | 0.6 | MAR above this = mouth wide open (yawn candidate). |
| `calibration.mar_margin` | 0.4 | Personal yawn threshold = max(mar_threshold, resting MAR + margin). |
| `mouth.yawn_min_duration` | 1.5 | Mouth must stay open this long to count as a yawn (talking is shorter). |
| `head.pitch_down_threshold` | 18 | Head pitched down more than this vs. calibrated neutral = head drop. |
| `head.nod_min_duration` | 0.4 | Head dips shorter than this are ignored (bumps, glances). |
| `head.head_drop_duration_threshold` | 2.0 | Sustained head drop that on its own raises the alarm. |
| `head.yaw_distraction_threshold` | 35 | Beyond this yaw the state is LOOKING AWAY (never drowsiness). |
| `face_loss.grace_period` | 1.0 | Face dropouts shorter than this are ignored. |
| `face_loss.warn_after` | 5.0 | Face missing this long -> DRIVER NOT VISIBLE warning (no drowsiness alarm). |
| `scoring.threshold` | 100 | Total drowsiness score at/above which the alarm switches on. |
| `scoring.release_threshold` | 40 | Acute score must fall below this before the alarm can stop. |
| `scoring.cumulative_cap` | 60 | Cap on fatigue-history score so history alone never reaches the threshold. |
| `alarm.min_duration` | 3.0 | Once on, the alarm sounds at least this long. |
| `alarm.release_time` | 1.5 | Eyes open / head up this long before the alarm stops. |

## 4. Results

| Metric | Result |
|---|---|
| False alarms/hour | Not yet measured — real labelled recordings required |
| Missed events | Not yet measured — real labelled recordings required |
| Alarm latency | Not yet measured — real labelled recordings required |
| Face/measurement loss rate | Not yet measured — real labelled recordings required |
| Processing FPS | Not yet measured — real labelled recordings required |

## 5. Breakdown by condition

Not yet measured — real labelled recordings required.

## 6. Evaluation environment

No evaluation run yet. The machine, Python and package versions are recorded automatically with every run (`run.json`).

## 7. Limitations

**Always applicable (protocol/framework design):**

- Drowsiness behaviours are **acted** following the recording script (instructed eye closures, yawns, head drops). Posed events can differ from genuine fatigue, so results describe detection of the scripted behaviours, not validated drowsiness detection.
- Recordings are made at a desk or in a parked vehicle (the project forbids testing while driving), so vehicle vibration, changing daylight and real driving attention demands are not represented.
- Ground truth is **manually annotated**; event boundaries are limited by frame rate and annotator judgement. The framework does not compute inter-annotator agreement.
- `alarm_expected` encodes the system's intended behaviour (alarm for prolonged eye closure or sustained head drop). Yawning alone is by design not expected to raise the alarm, so the evaluation does not judge whether yawning-only fatigue *should* be alarmed.
- Alarm latency is measured to the software alarm decision (`alarm_on`), not to physical buzzer/speaker output; audio, GPIO and serial delays are excluded (see issue #3).
- Processing FPS is measured offline on video files without the dashboard or live camera capture, on the machine that ran the evaluation. It must be re-measured on target hardware (e.g. Raspberry Pi) before being quoted for that hardware.
- The evaluation adapter (`evaluation/core/pipeline.py`) mirrors the per-frame loop of `main.py` using the same production modules. A drift test guards this, but a change to `main.py` must be mirrored manually.

**Specific to this dataset/run:**

- No real labelled recordings have been evaluated yet; no performance claims can be made from this report.

## 8. Reproduce

See `evaluation/README.md` → *Reproducing the evaluation*.
