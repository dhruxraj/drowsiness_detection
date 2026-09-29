"""Label and recording-manifest loading / validation (evaluation framework, issue #4).

All data below are UNIT-TEST FIXTURES - synthetic, not real recordings or labels.
"""
import unittest

import yaml

from evaluation.core import eval_config
from evaluation.core.dataset import (LABEL_COLUMNS, RECORDING_COLUMNS, DatasetError, load_labels,
                                     load_recordings)
from tests.evaluation_fixtures import FixtureWorkspace, label_row, manifest_row, write_csv


class _Base(unittest.TestCase):
    def setUp(self):
        self.ws = FixtureWorkspace()
        self.cfg = self.ws.cfg
        self.voc = self.cfg["vocabulary"]
        self.gt = self.cfg["ground_truth"]
        self.man = self.ws.root / "final_recordings.csv"
        self.lab = self.ws.root / "final_labels.csv"

    def tearDown(self):
        self.ws.cleanup()

    def recs(self, *rows, split="final", **kw):
        write_csv(self.man, RECORDING_COLUMNS, rows)
        return load_recordings(self.man, split, self.voc, **kw)

    def labels(self, recordings, *rows):
        write_csv(self.lab, LABEL_COLUMNS, rows)
        return load_labels(self.lab, recordings, self.voc, self.gt)

    def assertErrorContains(self, exc: DatasetError, *fragments):
        text = "\n".join(exc.errors)
        for f in fragments:
            self.assertIn(f, text)


class RecordingManifestTests(_Base):
    def test_valid_manifest_is_loaded_with_types(self):
        recs, warnings = self.recs(manifest_row("R1", "S1", "final", camera_distance_cm=""),
                                   manifest_row("R2", "S2", "final", glasses="clear",
                                                resolution="1280 x 720"))
        self.assertEqual([r.recording_id for r in recs], ["R1", "R2"])
        self.assertEqual(recs[0].duration_s, 120.0)
        self.assertEqual(recs[0].source_fps, 30.0)
        self.assertIsNone(recs[0].camera_distance_cm)
        self.assertEqual((recs[1].width, recs[1].height), (1280, 720))
        self.assertEqual(recs[1].glasses, "clear")
        # missing sha256 is allowed but warned about (needed for duplicate-content detection)
        self.assertTrue(any("sha256 is empty" in w for w in warnings))

    def test_header_only_manifest_is_empty_not_an_error(self):
        recs, _ = self.recs()
        self.assertEqual(recs, [])

    def test_missing_required_column(self):
        path = self.ws.root / "broken.csv"
        path.write_text("recording_id,subject_id\nR1,S1\n", encoding="utf-8")
        with self.assertRaises(DatasetError) as cm:
            load_recordings(path, "final", self.voc)
        self.assertErrorContains(cm.exception, "missing required column")

    def test_all_problems_are_reported_together(self):
        with self.assertRaises(DatasetError) as cm:
            self.recs(
                manifest_row("R1", "S1", "tuning"),                       # wrong split
                manifest_row("R2", "S2", "final", consent="no"),          # no consent
                manifest_row("R3", "S3", "final", lighting="disco"),      # not in vocabulary
                manifest_row("R4", "S4", "final", file="../other/R4.mp4"),  # escapes dataset dir
                manifest_row("R5", "S5", "final", file="/abs/R5.mp4"),
                manifest_row("R6", "S6", "final", calibration_end_s=500),  # >= duration
                manifest_row("R7", "S7", "final", resolution="640by480"),
                manifest_row("R8", "Jane Doe", "final"),                   # not a pseudonymous ID
                manifest_row("R9", "S9", "final", source_fps=0),
                manifest_row("r1", "S10", "final"),                        # duplicate (case-insensitive)
                manifest_row("R11", "S11", "final", sha256="xyz"),
            )
        self.assertErrorContains(
            cm.exception, "tuning and final data must stay separate", "consent must be 'yes'",
            "'disco' is not in the vocabulary", "must be a relative path", "calibration_end_s must be <",
            "resolution must look like", "pseudonymous", "source_fps must be > 0",
            "duplicate recording_id", "sha256 must be 64 hex")
        self.assertGreaterEqual(len(cm.exception.errors), 11)

    def test_check_files_reports_missing_video(self):
        with self.assertRaises(DatasetError) as cm:
            self.recs(manifest_row("R1", "S1", "final"), dataset_dir=self.ws.root / "dataset" / "final",
                      check_files=True)
        self.assertErrorContains(cm.exception, "recording file not found")
        self.ws.touch_videos("final", ["R1"])
        recs, _ = self.recs(manifest_row("R1", "S1", "final"),
                            dataset_dir=self.ws.root / "dataset" / "final", check_files=True)
        self.assertEqual(len(recs), 1)


class LabelTests(_Base):
    def setUp(self):
        super().setUp()
        self.recordings, _ = self.recs(manifest_row("R1", "S1", "final"),
                                       manifest_row("R2", "S2", "final"))

    def test_valid_labels_sorted_and_typed(self):
        labels, warnings = self.labels(
            self.recordings,
            label_row("R1", "e2", "yawn", 40, 43, "no"),
            label_row("R1", "e1", "eye_closure", 20, 24, "yes"),
            label_row("R1", "e3", "face_loss", 60, 62, "no"),
            label_row("R1", "e4", "head_drop", 70, 74, "YES"),  # case-insensitive
            label_row("R1", "e5", "blink", 10, 30, "no"),
        )
        self.assertEqual([e.event_id for e in labels["R1"]], ["e5", "e1", "e2", "e3", "e4"])
        e1 = labels["R1"][1]
        self.assertTrue(e1.is_positive)
        self.assertEqual(e1.duration_s, 4.0)
        self.assertEqual(labels["R1"][4].alarm_expected, "yes")
        # R2 has no labels: evaluated as a non-event recording, with a warning
        self.assertEqual(labels["R2"], [])
        self.assertTrue(any("'R2' has no labels" in w for w in warnings))

    def test_all_event_types_of_the_issue_are_supported(self):
        rows = [label_row("R1", f"e{i}", t, 10 + 10 * i, 12 + 10 * i, "no")
                for i, t in enumerate(("normal", "blink", "eye_closure", "yawn", "head_movement",
                                       "face_loss", "talking", "head_drop", "other"))]
        labels, _ = self.labels(self.recordings, *rows)
        self.assertEqual(len(labels["R1"]), 9)

    def test_invalid_labels_are_rejected_with_all_reasons(self):
        with self.assertRaises(DatasetError) as cm:
            self.labels(
                self.recordings,
                label_row("R9", "a", "yawn", 20, 22, "no"),               # unknown recording
                label_row("R1", "b", "sneeze", 20, 22, "no"),             # unknown event type
                label_row("R1", "c", "yawn", 20, 23, "yes"),              # yawn may not require alarm
                label_row("R1", "d", "blink", 30, 29, "no"),              # end <= start
                label_row("R1", "e", "blink", 110, 130, "no"),            # beyond duration (120 s)
                label_row("R1", "f", "eye_closure", 5, 9, "yes"),         # inside calibration (8 s)
                label_row("R1", "g", "blink", "abc", 5, "no"),            # not a number
                label_row("R1", "h", "blink", 40, 41, "maybe"),           # bad alarm_expected
                label_row("R1", "h", "blink", 42, 43, "no"),              # duplicate event_id
                label_row("R1", "bad id", "blink", 44, 45, "no"),         # invalid event_id
            )
        self.assertErrorContains(
            cm.exception, "not in the recordings manifest", "unknown event_type 'sneeze'",
            "not allowed for event_type 'yawn'", "end_s must be greater than start_s",
            "beyond the recording duration", "inside the calibration period",
            "start_s must be a number", "alarm_expected must be one of", "duplicate event_id",
            "event_id 'bad id'")

    def test_overlapping_alarm_expected_events_are_rejected(self):
        with self.assertRaises(DatasetError) as cm:
            self.labels(self.recordings,
                        label_row("R1", "a", "eye_closure", 20, 25, "yes"),
                        label_row("R1", "b", "head_drop", 24, 30, "yes"))
        self.assertErrorContains(cm.exception, "overlap")

    def test_protocol_definition_warnings(self):
        _, warnings = self.labels(
            self.recordings,
            label_row("R1", "a", "eye_closure", 20, 21.5, "yes"),   # shorter than 3 s protocol
            label_row("R1", "b", "eye_closure", 30, 32, "no"),      # 2 s marked 'no'
            label_row("R1", "c", "head_drop", 40, 41, "yes"))
        text = "\n".join(warnings)
        self.assertIn("R1/a: eye_closure marked alarm_expected=yes but lasts only 1.50s", text)
        self.assertIn("R1/b: eye_closure of 2.00s marked alarm_expected=no", text)
        self.assertIn("R1/c: head_drop marked alarm_expected=yes", text)


class EvalConfigTests(unittest.TestCase):
    def test_unquoted_yaml_yes_no_are_normalised(self):
        cfg = eval_config.from_dict(yaml.safe_load(
            "vocabulary:\n  event_types:\n    eye_closure: [yes, no, ignore]\n"))
        self.assertEqual(cfg["vocabulary"]["event_types"]["eye_closure"], ["yes", "no", "ignore"])

    def test_repository_config_is_valid(self):
        cfg = eval_config.load()
        self.assertIn("eye_closure", cfg["vocabulary"]["event_types"])
        self.assertGreaterEqual(cfg["matching"]["pre_tolerance_s"], 0)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(eval_config.EvalConfigError):
            eval_config.from_dict({"matching": {"pre_tolerance_s": -1}})

    def test_measurement_parameters_cannot_be_tuned_by_replay(self):
        with self.assertRaises(eval_config.EvalConfigError) as cm:
            eval_config.from_dict({"tuning": {"search_space": {"face_mesh.min_detection_confidence": [0.4]}}})
        self.assertIn("cannot be tuned by trace replay", str(cm.exception))

    def test_invalid_alarm_expected_vocabulary_rejected(self):
        with self.assertRaises(eval_config.EvalConfigError):
            eval_config.from_dict({"vocabulary": {"event_types": {"x": ["sometimes"]}}})


if __name__ == "__main__":
    unittest.main()
