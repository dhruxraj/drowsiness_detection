"""Result files and command-line workflow (evaluation framework, issue #4).

All data below are UNIT-TEST FIXTURES - synthetic, not real recordings or labels. The
workflow tests run inside a temporary workspace; the repository's evaluation/labels and
evaluation/results are never read or written.
"""
import contextlib
import csv
import importlib.util
import io
import json
import unittest
from pathlib import Path

import yaml

from evaluation.core import report as rp
from evaluation.core.metrics import aggregate, breakdowns, dataset_summary, evaluate_recording
from evaluation.core.trace import OK, RecordingRun, write_trace
from tests.evaluation_fixtures import (FIXTURE_SYSTEM_CONFIG, FixtureWorkspace, eval_cfg, event,
                                       label_row, make_trace, manifest_row, recording)

SCRIPTS = Path(__file__).resolve().parents[1] / "evaluation" / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"eval_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_script(name, *argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = load_script(name).main(list(argv))
    return code, out.getvalue()


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class PlaceholderTests(unittest.TestCase):
    def setUp(self):
        self.ws = FixtureWorkspace()

    def tearDown(self):
        self.ws.cleanup()

    def test_nothing_measured_is_stated_explicitly(self):
        out = self.ws.root / "results"
        cfg = eval_cfg()
        ds = dataset_summary([], {})
        rp.write_metrics_csv(out / "metrics.csv", None)
        rp.write_breakdown_csv(out / "breakdown.csv", None)
        text = rp.render_report(context={"generated_at": "fixture", "split": "final"}, dataset=ds,
                                summary=None, breakdowns=None, system_cfg=FIXTURE_SYSTEM_CONFIG,
                                eval_cfg=cfg, frozen=None)
        rows = read_rows(out / "metrics.csv")
        self.assertTrue(rows)
        self.assertTrue(all(r["status"] == rp.TBD_CSV and r["value"] == "" for r in rows))
        names = {r["metric"] for r in rows}
        for required in ("false_alarms_per_hour", "missed_events", "latency_ms_mean", "latency_ms_median",
                         "latency_ms_min", "latency_ms_max", "measurement_unavailable_rate_pct",
                         "processing_fps_mean"):
            self.assertIn(required, names)
        self.assertTrue(all(r["status"] == rp.TBD_CSV for r in read_rows(out / "breakdown.csv")))
        results = text.split("## 4. Results")[1].split("## 5.")[0]
        for metric in ("False alarms/hour", "Missed events", "Alarm latency", "Face/measurement loss rate",
                       "Processing FPS"):
            self.assertIn(f"| {metric} | {rp.NOT_MEASURED} |", results)
        self.assertIn("Thresholds are not frozen yet", text)
        self.assertIn("No real labelled recordings have been evaluated yet", text)

    def test_measured_summary_is_written_with_units_and_sample_sizes(self):
        cfg = eval_cfg()
        rec = recording("FIXTURE_R01", "FIXTURE_S01", duration_s=120)
        labels = [event("e1", "eye_closure", 50, 54, "yes")]
        frames = make_trace(120, 4, 10, alarms=[(51.0, 56.0), (80.0, 83.0)])
        ev = evaluate_recording(rec, labels, RecordingRun("FIXTURE_R01", OK, frames=frames), cfg,
                                FIXTURE_SYSTEM_CONFIG["face_loss"])
        s, bd = aggregate([ev]), breakdowns([ev], cfg)
        out = self.ws.root / "results"
        rp.write_metrics_csv(out / "metrics.csv", s)
        rp.write_breakdown_csv(out / "breakdown.csv", bd)
        rows = {r["metric"]: r for r in read_rows(out / "metrics.csv")}
        self.assertEqual(rows["missed_events"]["value"], "0")
        self.assertEqual(rows["missed_events"]["status"], "measured")
        self.assertEqual(rows["latency_ms_mean"]["value"], "1000.0")
        self.assertEqual(rows["latency_ms_mean"]["unit"], "ms")
        self.assertEqual(rows["false_alarms"]["value"], "1")
        self.assertEqual(rows["processing_fps_mean"]["unit"], "frames/s")
        bd_rows = read_rows(out / "breakdown.csv")
        self.assertTrue(all(r["status"] in ("insufficient_data", "undefined") for r in bd_rows))
        text = rp.render_report(context={"generated_at": "fixture", "split": "tuning", "official": False,
                                         "not_official_reason": "fixture"},
                                dataset=dataset_summary([rec], {"FIXTURE_R01": labels}), summary=s,
                                breakdowns=bd, system_cfg=FIXTURE_SYSTEM_CONFIG, eval_cfg=cfg, frozen=None)
        self.assertIn("NOT an official final result", text)
        self.assertIn("mean 1000 ms, median 1000 ms, min 1000 ms, max 1000 ms (n=1)", text)
        self.assertIn("Limited number of subjects (1).", text)
        self.assertNotIn(rp.NOT_MEASURED, text.split("## 4. Results")[1].split("## 5.")[0])


class WorkflowTests(unittest.TestCase):
    """End-to-end CLI checks on a synthetic run directory (no MediaPipe needed)."""

    def setUp(self):
        self.ws = FixtureWorkspace()
        self.cfg_arg = ["--config", str(self.ws.config_path)]
        self.ws.write_split("tuning", [manifest_row("T1", "S1", "tuning")],
                            [label_row("T1", "e1", "eye_closure", 20, 24, "yes")])
        self.ws.write_split("final", [manifest_row("F1", "S2", "final"), manifest_row("F2", "S3", "final")],
                            [label_row("F1", "e1", "eye_closure", 20, 24, "yes"),
                             label_row("F1", "e2", "yawn", 40, 43, "no"),
                             label_row("F2", "e1", "head_drop", 60, 64, "yes")])

    def tearDown(self):
        self.ws.cleanup()

    def _fake_run(self, split="final", ids=("F1", "F2")):
        """A run directory as evaluate.py would write it, with synthetic traces."""
        run_dir = self.ws.root / "runs" / f"fixture_{split}"
        system_cfg = FIXTURE_SYSTEM_CONFIG
        from evaluation.core import thresholds as th
        meta = {"run_id": f"fixture_{split}", "split": split, "subset": False,
                "detection_fingerprint": th.fingerprint(th.detection_snapshot(system_cfg)),
                "frozen_detection_fingerprint": None, "recordings": {}}
        alarms = {"F1": [(21.0, 26.0)], "F2": [(90.0, 93.0)], "T1": [(22.0, 26.0)]}
        for rid in ids:
            run = RecordingRun(rid, OK, frames=make_trace(120, 4, 8, alarms=alarms[rid]))
            write_trace(run_dir / "traces" / f"{rid}.csv", run.frames)
            meta["recordings"][rid] = run.meta()
        (run_dir / "run.json").write_text(json.dumps(meta), encoding="utf-8")
        return run_dir

    def test_validate_dataset_detects_leakage(self):
        code, _ = run_script("validate_dataset", *self.cfg_arg)
        self.assertEqual(code, 0)
        self.ws.write_split("final", [manifest_row("F1", "S1", "final")], [])   # S1 is a tuning subject
        code, out = run_script("validate_dataset", *self.cfg_arg)
        self.assertEqual(code, 2)
        self.assertIn("TUNING/FINAL LEAKAGE", out)

    def test_evaluate_final_refuses_without_frozen_thresholds(self):
        self.ws.touch_videos("tuning", ["T1"])
        self.ws.touch_videos("final", ["F1", "F2"])
        code, out = run_script("evaluate", "--split", "final", *self.cfg_arg)
        self.assertEqual(code, 3)
        self.assertIn("thresholds are not frozen", out)

    def test_evaluate_rejects_missing_video_files(self):
        code, out = run_script("evaluate", "--split", "final", *self.cfg_arg)
        self.assertEqual(code, 2)
        self.assertIn("recording file not found", out)

    def test_freeze_then_recompute_and_publish_official_result(self):
        code, out = run_script("freeze_thresholds", *self.cfg_arg, "--selection-method", "unit-test fixture")
        self.assertEqual(code, 0, out)
        frozen = yaml.safe_load((self.ws.root / "frozen_thresholds.yaml").read_text(encoding="utf-8"))
        self.assertEqual(frozen["status"], "frozen")
        self.assertEqual(frozen["tuning_recordings"], ["T1"])
        self.assertEqual(frozen["tuning_subjects"], ["S1"])
        code, out = run_script("freeze_thresholds", *self.cfg_arg, "--selection-method", "again")
        self.assertEqual(code, 1)                                 # refuses to silently re-freeze

        run_dir = self._fake_run()
        code, out = run_script("calculate_metrics", *self.cfg_arg, "--run-dir", str(run_dir), "--publish")
        self.assertEqual(code, 0, out)
        results = self.ws.root / "results"
        rows = {r["metric"]: r for r in read_rows(results / "metrics.csv")}
        self.assertEqual(rows["missed_events"]["value"], "1")       # F2/e1 head_drop has no alarm
        self.assertEqual(rows["false_alarms"]["value"], "1")        # F2 alarm at 90 s
        self.assertEqual(rows["latency_ms_median"]["value"], "1000.0")
        summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["context"]["official"])
        matches = read_rows(run_dir / "matches.csv")
        self.assertEqual(sorted((m["recording_id"], m["outcome"]) for m in matches),
                         [("F1", "detected"), ("F2", "missed")])
        report = (results / "report.md").read_text(encoding="utf-8")
        self.assertIn("final evaluation on the held-out FINAL split with frozen thresholds", report)
        self.assertIn("| tuning (threshold selection, from the frozen record) | S1 | T1 |", report)

    def test_results_not_published_when_config_changed_after_freezing(self):
        run_script("freeze_thresholds", *self.cfg_arg, "--selection-method", "unit-test fixture")
        changed = dict(FIXTURE_SYSTEM_CONFIG, eyes={"ear_threshold": 0.25, "closure_duration_threshold": 1.8})
        self.ws.system_config.write_text(yaml.safe_dump(changed), encoding="utf-8")
        run_dir = self._fake_run()
        code, out = run_script("calculate_metrics", *self.cfg_arg, "--run-dir", str(run_dir), "--publish")
        self.assertEqual(code, 3)
        self.assertIn("Not published", out)
        self.assertFalse((self.ws.root / "results" / "summary.json").exists())

    def test_tuning_run_is_never_published(self):
        run_script("freeze_thresholds", *self.cfg_arg, "--selection-method", "unit-test fixture")
        run_dir = self._fake_run(split="tuning", ids=("T1",))
        code, out = run_script("calculate_metrics", *self.cfg_arg, "--run-dir", str(run_dir), "--publish")
        self.assertEqual(code, 3)
        self.assertIn("tuning-split run", out)
        self.assertTrue((run_dir / "metrics.csv").is_file())       # metrics still computed in the run dir

    def test_placeholder_removes_stale_summary(self):
        results = self.ws.root / "results"
        (results / "summary.json").write_text("{}", encoding="utf-8")
        code, _ = run_script("calculate_metrics", *self.cfg_arg, "--placeholder")
        self.assertEqual(code, 0)
        self.assertFalse((results / "summary.json").exists())
        self.assertIn(rp.NOT_MEASURED, (results / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
