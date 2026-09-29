"""Loading of the tuning / final splits and enforcement of their separation.

The issue requires that *subjects and recordings* used for threshold tuning are kept
separate from those used for the final evaluation. ``check_separation`` therefore
rejects a dataset when the two splits share any of:

* a recording_id (case-insensitive),
* a subject_id (case-insensitive) - subject-level separation, not only recording-level,
* a recording file (same resolved path), or
* identical file content (same SHA-256, when hashes are filled in).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import eval_config
from .dataset import DatasetError, LabelEvent, Recording, load_labels, load_recordings


class SplitLeakageError(DatasetError):
    """Tuning and final data overlap."""


@dataclass
class SplitData:
    name: str
    dataset_dir: Path
    recordings_path: Path
    labels_path: Path
    recordings: List[Recording] = field(default_factory=list)
    labels: Dict[str, List[LabelEvent]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def video_path(self, rec: Recording) -> Path:
        return self.dataset_dir / rec.file


def split_paths(cfg: Dict, split: str) -> Dict[str, Path]:
    p = cfg["paths"]["splits"][split]
    return {k: eval_config.resolve(v) for k, v in p.items()}


def load_split(cfg: Dict, split: str, *, with_labels: bool = True, check_files: bool = False,
               recordings_path: Optional[Path] = None, labels_path: Optional[Path] = None,
               dataset_dir: Optional[Path] = None) -> SplitData:
    paths = split_paths(cfg, split)
    data = SplitData(
        name=split,
        dataset_dir=Path(dataset_dir) if dataset_dir else paths["dataset_dir"],
        recordings_path=Path(recordings_path) if recordings_path else paths["recordings"],
        labels_path=Path(labels_path) if labels_path else paths["labels"],
    )
    data.recordings, w1 = load_recordings(data.recordings_path, split, cfg["vocabulary"],
                                          data.dataset_dir, check_files=check_files)
    data.warnings.extend(w1)
    if with_labels:
        data.labels, w2 = load_labels(data.labels_path, data.recordings, cfg["vocabulary"],
                                      cfg["ground_truth"])
        data.warnings.extend(w2)
    return data


def check_separation(tuning: SplitData, final: SplitData) -> List[str]:
    """Return a list of leakage problems (empty list = splits are properly separated)."""
    problems: List[str] = []

    def overlap(values_a, values_b):
        return sorted(set(values_a) & set(values_b))

    ids = overlap((r.recording_id.lower() for r in tuning.recordings),
                  (r.recording_id.lower() for r in final.recordings))
    if ids:
        problems.append(f"recording_id(s) present in both tuning and final splits: {ids}")
    subjects = overlap((r.subject_id.lower() for r in tuning.recordings),
                       (r.subject_id.lower() for r in final.recordings))
    if subjects:
        problems.append(f"subject_id(s) present in both tuning and final splits: {subjects} - "
                        "the same person must not be used for tuning and final evaluation")
    files = overlap((str(tuning.video_path(r).resolve()) for r in tuning.recordings),
                    (str(final.video_path(r).resolve()) for r in final.recordings))
    if files:
        problems.append(f"the same recording file is referenced by both splits: {files}")
    hashes = overlap((r.sha256 for r in tuning.recordings if r.sha256),
                     (r.sha256 for r in final.recordings if r.sha256))
    if hashes:
        problems.append(f"identical file content (sha256) in both splits: {hashes}")
    return problems


def assert_separated(tuning: SplitData, final: SplitData) -> None:
    problems = check_separation(tuning, final)
    if problems:
        raise SplitLeakageError(problems)


def load_both(cfg: Dict, *, check_files: bool = False, with_labels: Sequence[str] = ("tuning", "final"),
              overrides: Optional[Dict[str, Dict[str, Path]]] = None) -> Dict[str, SplitData]:
    """Load both splits (labels only where requested) and verify their separation."""
    overrides = overrides or {}
    out = {}
    for split in ("tuning", "final"):
        out[split] = load_split(cfg, split, with_labels=split in with_labels,
                                check_files=check_files, **overrides.get(split, {}))
    assert_separated(out["tuning"], out["final"])
    return out
