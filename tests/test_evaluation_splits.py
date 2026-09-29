"""Tuning / final dataset separation and threshold freezing (evaluation framework, issue #4).

All data below are UNIT-TEST FIXTURES - synthetic, not real recordings or labels.
"""
import copy
import unittest

from evaluation.core import thresholds as th
from evaluation.core.splits import SplitLeakageError, check_separation, load_both, load_split
from evaluation.core.workflow import official_status
from tests.evaluation_fixtures import FIXTURE_SYSTEM_CONFIG, FixtureWorkspace, manifest_row


class SeparationTests(unittest.TestCase):
    def setUp(self):
        self.ws = FixtureWorkspace()

    def tearDown(self):
        self.ws.cleanup()

    def _write(self, tuning_rows, final_rows):
        self.ws.write_split("tuning", tuning_rows, [])
        self.ws.write_split("final", final_rows, [])

    def test_disjoint_subjects_and_recordings_are_accepted(self):
        self._write([manifest_row("T1", "S1", "tuning"), manifest_row("T2", "S2", "tuning")],
                    [manifest_row("F1", "S3", "final")])
        splits = load_both(self.ws.cfg)
        self.assertEqual(check_separation(splits["tuning"], splits["final"]), [])

    def test_same_recording_id_is_rejected(self):
        self._write([manifest_row("R1", "S1", "tuning")], [manifest_row("r1", "S2", "final")])
        with self.assertRaises(SplitLeakageError) as cm:
            load_both(self.ws.cfg)
        self.assertIn("recording_id(s) present in both", "\n".join(cm.exception.errors))

    def test_same_subject_is_rejected_even_with_different_recordings(self):
        self._write([manifest_row("T1", "S1", "tuning")], [manifest_row("F1", "s1", "final")])
        with self.assertRaises(SplitLeakageError) as cm:
            load_both(self.ws.cfg)
        self.assertIn("subject_id(s) present in both", "\n".join(cm.exception.errors))

    def test_same_file_content_is_rejected(self):
        digest = "a" * 64
        self._write([manifest_row("T1", "S1", "tuning", sha256=digest)],
                    [manifest_row("F1", "S2", "final", sha256=digest.upper())])
        with self.assertRaises(SplitLeakageError) as cm:
            load_both(self.ws.cfg)
        self.assertIn("identical file content", "\n".join(cm.exception.errors))

    def test_same_file_path_is_rejected(self):
        self._write([manifest_row("T1", "S1", "tuning", file="shared.mp4")],
                    [manifest_row("F1", "S2", "final", file="shared.mp4")])
        cfg = self.ws.cfg
        shared_dir = self.ws.root / "dataset" / "tuning"
        tuning = load_split(cfg, "tuning", dataset_dir=shared_dir)
        final = load_split(cfg, "final", dataset_dir=shared_dir)  # misconfigured: same folder
        problems = check_separation(tuning, final)
        self.assertTrue(any("same recording file" in p for p in problems))

    def test_final_row_in_tuning_manifest_is_rejected(self):
        self._write([manifest_row("T1", "S1", "final")], [])
        with self.assertRaises(Exception) as cm:
            load_both(self.ws.cfg)
        self.assertIn("tuning and final data must stay separate", str(cm.exception))


class FrozenThresholdTests(unittest.TestCase):
    def setUp(self):
        self.ws = FixtureWorkspace()
        self.cfg = self.ws.cfg
        self.sys = copy.deepcopy(FIXTURE_SYSTEM_CONFIG)
        self.frozen = th.build_frozen_record(
            self.sys, self.cfg, tuning_recording_ids=["T1", "T2"], tuning_subject_ids=["S1", "S2", "S1"],
            tuning_manifest_sha256=None, selection_method="unit-test fixture", tuning_run=None,
            git_commit=None, frozen_at="2000-01-01T00:00:00+00:00", notes="", final_results_existed=False)

    def tearDown(self):
        self.ws.cleanup()

    def test_record_contents(self):
        self.assertEqual(self.frozen["status"], th.FROZEN)
        self.assertEqual(self.frozen["tuning_subjects"], ["S1", "S2"])
        self.assertEqual(self.frozen["detection_config"]["eyes"]["closure_duration_threshold"], 1.8)
        self.assertEqual(self.frozen["protocol"]["matching"], self.cfg["matching"])

    def test_not_frozen_is_a_problem(self):
        self.assertTrue(th.verify_frozen(None, self.sys, self.cfg, ["F1"], ["S3"]))
        self.assertTrue(th.verify_frozen({"status": th.NOT_FROZEN}, self.sys, self.cfg, ["F1"], ["S3"]))

    def test_unchanged_config_passes(self):
        self.assertEqual(th.verify_frozen(self.frozen, self.sys, self.cfg, ["F1"], ["S3"]), [])

    def test_changed_detection_threshold_fails(self):
        changed = copy.deepcopy(self.sys)
        changed["eyes"]["closure_duration_threshold"] = 1.5
        problems = th.verify_frozen(self.frozen, changed, self.cfg, ["F1"], ["S3"])
        self.assertTrue(any("eyes" in p for p in problems))

    def test_non_detection_setting_does_not_matter(self):
        changed = copy.deepcopy(self.sys)
        changed["alarm"]["backends"] = ["gpio"]           # hardware choice, not a threshold
        changed["camera"] = {"source": 1}
        self.assertEqual(th.verify_frozen(self.frozen, changed, self.cfg, ["F1"], ["S3"]), [])

    def test_changed_matching_protocol_fails(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["matching"]["post_tolerance_s"] = 5.0
        problems = th.verify_frozen(self.frozen, self.sys, cfg, ["F1"], ["S3"])
        self.assertTrue(any("protocol" in p for p in problems))

    def test_tuning_subject_in_final_split_fails(self):
        problems = th.verify_frozen(self.frozen, self.sys, self.cfg, ["F1", "t2"], ["s1"])
        self.assertTrue(any("final recordings were used for tuning" in p for p in problems))
        self.assertTrue(any("final subjects were used for tuning" in p for p in problems))

    def test_key_thresholds_have_rationale(self):
        rows = th.key_thresholds(self.sys)
        self.assertTrue(all(r["rationale"] for r in rows))
        self.assertIn({"key": "eyes.closure_duration_threshold", "value": 1.8,
                       "rationale": th.THRESHOLD_RATIONALE["eyes.closure_duration_threshold"]}, rows)


class OfficialStatusTests(unittest.TestCase):
    def setUp(self):
        self.ws = FixtureWorkspace()
        self.ws.write_split("final", [manifest_row("F1", "S3", "final"), manifest_row("F2", "S4", "final")], [])
        self.final = load_split(self.ws.cfg, "final", with_labels=False)

    def tearDown(self):
        self.ws.cleanup()

    def meta(self, **kw):
        m = {"split": "final", "subset": False, "recordings": {"F1": {}, "F2": {}},
             "detection_fingerprint": "x", "frozen_detection_fingerprint": "x"}
        m.update(kw)
        return m

    def test_complete_frozen_final_run_is_official(self):
        self.assertEqual(official_status(self.meta(), self.final, []), (True, ""))

    def test_tuning_run_is_never_official(self):
        self.assertFalse(official_status(self.meta(split="tuning"), self.final, [])[0])

    def test_unfrozen_subset_or_changed_config_is_not_official(self):
        self.assertFalse(official_status(self.meta(), self.final, ["not frozen"])[0])
        self.assertFalse(official_status(self.meta(subset=True), self.final, [])[0])
        self.assertFalse(official_status(self.meta(recordings={"F1": {}}), self.final, [])[0])
        self.assertFalse(official_status(self.meta(frozen_detection_fingerprint="y"), self.final, [])[0])
        self.assertFalse(official_status(self.meta(frame_step=2), self.final, [])[0])


if __name__ == "__main__":
    unittest.main()
