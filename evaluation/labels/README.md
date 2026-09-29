# Labels and recording manifests

| File | Split | Content |
|---|---|---|
| `tuning_recordings.csv` | tuning | one row per tuning recording (subject, conditions, camera, FPS, resolution, calibration end, consent) |
| `tuning_labels.csv` | tuning | labelled event intervals of the tuning recordings |
| `evaluation_recordings.csv` | final | one row per final-evaluation recording |
| `evaluation_labels.csv` | final | labelled event intervals of the final recordings |

The files currently contain **only the header rows** - no recordings have been labelled yet.
The column definitions, allowed values, labelling guide and consent rules are in
[`../dataset/README.md`](../dataset/README.md). Use pseudonymous IDs only; these files are
committed, the videos are not.

Check the files with `python evaluation/scripts/validate_dataset.py`.
