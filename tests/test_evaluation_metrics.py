"""Metric calculations (evaluation framework, issue #4).

All data below are UNIT-TEST FIXTURES - synthetic, not real recordings or labels. The
expected numbers are derived by hand from the fixture definitions in each test, so they
check the arithmetic of the metric code only; they are NOT experimental results.
"""
import unittest

from evaluation.core.metrics import (EXTENDED, SHORT_DROPOUT, TEMPORARY, aggregate, breakdowns,
                                     dataset_summary, evaluate_recording)
from evaluation.core.trace import FAILED_OPEN, NO_FACE, OK, RecordingRun
from tests.evaluation_fixtures import eval_cfg, event, make_trace, recording

FACE_LOSS_CFG = {"grace_period": 1.0, "warn_after": 5.0}


def scenario():
    """400 s fixture recording at 4 fps, labelled calibration period 0-10 s.

    Alarm-expected events: e1 eye_closure 100-104, e2 head_drop 320-324, e3 eye_closure 350-354
    Ignore event:           i1 eye_closure 200-202
    Context events:         y1 yawn 60-63, f1 face_loss 249-259
    Alarm episodes:         5-8 (calibration), 30-33 (FA), 101.5-106 (e1), 201-204 (ignored),
                            322-327 (e2), 380-383 (FA); e3 gets no alarm (missed)
    Face missing:           2-3 (calibration), 60-60.5 (short), 150-153 (temporary),
                            250-258 (extended, inside labelled face_loss)
    Invalid measurements:   300-301
    """
    rec = recording("FIXTURE_R01", "FIXTURE_S01")
    labels = [
        event("e1", "eye_closure", 100.0, 104.0, "yes"),
        event("e2", "head_drop", 320.0, 324.0, "yes"),
        event("e3", "eye_closure", 350.0, 354.0, "yes"),
        event("i1", "eye_closure", 200.0, 202.0, "ignore"),
        event("y1", "yawn", 60.0, 63.0, "no"),
        event("f1", "face_loss", 249.0, 259.0, "no"),
    ]
    frames = make_trace(
        400.0, 4.0, 10.0,
        alarms=[(5.0, 8.0), (30.0, 33.0), (101.5, 106.0), (201.0, 204.0), (322.0, 327.0), (380.0, 383.0)],
        face_missing=[(2.0, 3.0), (60.0, 60.5), (150.0, 153.0), (250.0, 258.0)],
        invalid=[(300.0, 301.0)], processing_ms=20.0, acquire_ms=5.0)
    return rec, labels, RecordingRun("FIXTURE_R01", OK, frames=frames)


class RecordingMetricTests(unittest.TestCase):
    def setUp(self):
        self.cfg = eval_cfg()
        rec, labels, run = scenario()
        self.ev = evaluate_recording(rec, labels, run, self.cfg, FACE_LOSS_CFG)
        self.s = aggregate([self.ev])

    # ---------------------------------------------------------------- false alarms
    def test_false_alarm_count_and_contexts(self):
        self.assertEqual(self.s["false_alarms"], 2)
        self.assertEqual(self.ev.fa_contexts, ["unlabelled (normal)", "unlabelled (normal)"])
        self.assertEqual(self.s["ignored_alarms"], 1)
        self.assertEqual(self.s["alarms_during_calibration"], 1)

    def test_false_alarms_per_hour_normalised_by_non_event_time(self):
        # evaluation window 10-400 s = 390 s
        # minus expanded positive windows 3 x (4 + 0.5 + 1.0) = 16.5 s
        # minus expanded ignore window (2 + 0.5 + 1.0) = 3.5 s  -> 370 s non-event time
        self.assertAlmostEqual(self.ev.evaluated_s, 386.5)
        self.assertAlmostEqual(self.ev.non_event_s, 370.0)
        self.assertAlmostEqual(self.s["false_alarms_per_hour"], 2 / 370.0 * 3600.0)
        self.assertAlmostEqual(self.s["false_alarms_per_evaluated_hour"], 2 / 386.5 * 3600.0)

    # ---------------------------------------------------------------- missed events
    def test_missed_events(self):
        self.assertEqual(self.s["positive_events"], 3)
        self.assertEqual(self.s["detected_events"], 2)
        self.assertEqual(self.s["missed_events"], 1)
        self.assertEqual(self.s["missed_event_ids"], ["FIXTURE_R01/e3"])
        self.assertAlmostEqual(self.s["detection_rate_pct"], 200.0 / 3.0)
        self.assertEqual(self.s["by_event_type"]["eye_closure"],
                         {"events": 2, "detected": 1, "missed": 1, "latency_median_ms": 1500.0})

    # ---------------------------------------------------------------- latency
    def test_latency_statistics_in_milliseconds(self):
        lat = self.s["latency_ms"]
        self.assertEqual(lat["n"], 2)
        self.assertAlmostEqual(lat["mean"], 1750.0)      # (1500 + 2000) / 2
        self.assertAlmostEqual(lat["median"], 1750.0)
        self.assertAlmostEqual(lat["min"], 1500.0)       # 101.5 - 100
        self.assertAlmostEqual(lat["max"], 2000.0)       # 322.0 - 320
        self.assertAlmostEqual(self.s["release_delay_ms"]["median"], 2500.0)  # (2000 + 3000) / 2

    # ---------------------------------------------------------------- face / measurement loss
    def test_face_and_measurement_loss_rates(self):
        # frames in the evaluation window: t = 10.00 ... 399.75 -> 1560 frames
        self.assertEqual(self.ev.frames_eval, 1560)
        # face missing: 60-60.5 (2) + 150-153 (12) + 250-258 (32) = 46; calibration 2-3 excluded
        self.assertEqual(self.ev.frames_face_missing, 46)
        self.assertEqual(self.ev.frames_invalid, 4)                  # 300-301
        self.assertEqual(self.ev.invalid_reasons, {"ear_invalid": 4})
        self.assertEqual(self.ev.frames_unavailable_expected, 32)    # inside labelled face_loss
        self.assertEqual(self.ev.frames_unavailable_unexpected, 18)
        self.assertAlmostEqual(self.s["face_missing_rate_pct"], 46 / 1560 * 100)
        self.assertAlmostEqual(self.s["invalid_measurement_rate_pct"], 4 / 1560 * 100)
        self.assertAlmostEqual(self.s["measurement_unavailable_rate_pct"], 50 / 1560 * 100)
        self.assertAlmostEqual(self.s["unexpected_unavailable_rate_pct"], 18 / 1560 * 100)
        self.assertAlmostEqual(self.s["measurement_unavailable_time_pct"], 12.5 / 390 * 100)

    def test_face_loss_episodes_are_classified(self):
        kinds = [(e.start_s, e.duration_s, e.kind, e.expected) for e in self.ev.loss_episodes]
        self.assertEqual(kinds, [(60.0, 0.5, SHORT_DROPOUT, False), (150.0, 3.0, TEMPORARY, False),
                                 (250.0, 8.0, EXTENDED, True)])
        self.assertEqual(self.s["face_loss_episodes"], {SHORT_DROPOUT: 1, TEMPORARY: 1, EXTENDED: 1})
        self.assertEqual(self.s["face_loss_episodes_unexpected"], 2)
        self.assertEqual(self.s["longest_face_loss_s"], 8.0)

    # ---------------------------------------------------------------- processing FPS
    def test_processing_fps_is_measured_not_source_fps(self):
        self.assertEqual(self.s["frames_processed"], 1600)            # all frames incl. calibration
        self.assertAlmostEqual(self.s["processing_fps_mean"], 50.0)   # 1000 / 20 ms
        self.assertAlmostEqual(self.s["processing_fps_median"], 50.0)
        self.assertAlmostEqual(self.s["processing_fps_p5"], 50.0)
        self.assertAlmostEqual(self.s["end_to_end_fps"], 40.0)        # 1000 / (20 + 5) ms
        self.assertEqual(self.s["source_fps_median"], 4.0)            # from the manifest
        self.assertAlmostEqual(self.s["source_fps_measured_median"], 4.0)
        self.assertEqual(self.ev.frame_shortfall_ratio, 0.0)


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = eval_cfg()

    def test_fps_pooled_over_frames_and_minimum_per_recording(self):
        fast = evaluate_recording(recording("A", "S1", duration_s=100), [],
                                  RecordingRun("A", OK, frames=make_trace(100, 4, 10, processing_ms=10.0)),
                                  self.cfg, FACE_LOSS_CFG)
        slow = evaluate_recording(recording("B", "S2", duration_s=100), [],
                                  RecordingRun("B", OK, frames=make_trace(100, 4, 10, processing_ms=40.0)),
                                  self.cfg, FACE_LOSS_CFG)
        s = aggregate([fast, slow])
        # pooled: 800 frames / (400 x 10 ms + 400 x 40 ms) = 800 / 20 s = 40 fps
        self.assertAlmostEqual(s["processing_fps_mean"], 800 / (400 * 0.010 + 400 * 0.040))
        self.assertAlmostEqual(s["processing_fps_median"], 1000 / 25.0)   # median of 10 ms / 40 ms
        self.assertAlmostEqual(s["processing_fps_min_recording"], 25.0)
        self.assertAlmostEqual(s["processing_fps_p5"], 25.0)

    def test_subsampled_run_uses_the_processed_frame_spacing(self):
        rec = recording("H", "S1", duration_s=100, source_fps=8.0)
        frames = make_trace(100, 4, 10)                      # every 2nd frame of an 8 fps video
        ev = evaluate_recording(rec, [], RecordingRun("H", OK, frames=frames, frame_step=2),
                                self.cfg, FACE_LOSS_CFG)
        self.assertEqual(ev.frame_shortfall_ratio, 0.0)
        self.assertEqual(ev.warnings, [])
        self.assertAlmostEqual(ev.time_eval_s, 90.0)
        full = evaluate_recording(rec, [], RecordingRun("H", OK, frames=frames), self.cfg, FACE_LOSS_CFG)
        self.assertAlmostEqual(full.frame_shortfall_ratio, 0.5)  # without frame_step: frames missing
        self.assertTrue(full.warnings)

    def test_failed_recording_is_excluded_but_reported(self):
        rec, labels, run = scenario()
        good = evaluate_recording(rec, labels, run, self.cfg, FACE_LOSS_CFG)
        bad_rec = recording("FIXTURE_R02", "FIXTURE_S02")
        bad = evaluate_recording(bad_rec, [event("x", "eye_closure", 50, 55, "yes", "FIXTURE_R02")],
                                 RecordingRun("FIXTURE_R02", FAILED_OPEN, error="cannot open"),
                                 self.cfg, FACE_LOSS_CFG)
        s = aggregate([good, bad])
        self.assertFalse(bad.included)
        self.assertEqual(s["recordings_failed"], 1)
        self.assertEqual(s["failed_recordings"], {"FIXTURE_R02": FAILED_OPEN})
        self.assertEqual(s["positive_events"], 3)                  # only from evaluated recordings
        self.assertEqual(s["positive_events_not_evaluated"], 1)
        self.assertEqual(s["subjects"], 2)
        self.assertEqual(s["subjects_evaluated"], 1)

    def test_recording_where_face_is_never_found_counts_as_misses(self):
        rec = recording("N", "S9", duration_s=60)
        frames = make_trace(60, 4, 10, face_missing=[(0.0, 60.0)])
        ev = evaluate_recording(rec, [event("p", "eye_closure", 20, 25, "yes", "N")],
                                RecordingRun("N", NO_FACE, frames=frames), self.cfg, FACE_LOSS_CFG)
        s = aggregate([ev])
        self.assertTrue(ev.included)
        self.assertEqual((s["missed_events"], s["recordings_no_face"]), (1, 1))
        self.assertEqual(s["missed_while_face_unavailable"], 1)
        self.assertAlmostEqual(s["face_missing_rate_pct"], 100.0)

    def test_nothing_measured_yields_no_numbers(self):
        s = aggregate([])
        self.assertFalse(s["measured"])
        for key in ("false_alarms_per_hour", "missed_events", "processing_fps_mean",
                    "measurement_unavailable_rate_pct"):
            self.assertIsNone(s[key], key)
        self.assertIsNone(s["latency_ms"]["mean"])

    def test_no_detections_gives_undefined_latency(self):
        rec = recording("Z", "S1", duration_s=60)
        ev = evaluate_recording(rec, [event("p", "eye_closure", 20, 25, "yes", "Z")],
                                RecordingRun("Z", OK, frames=make_trace(60, 4, 10)), self.cfg, FACE_LOSS_CFG)
        s = aggregate([ev])
        self.assertEqual(s["latency_ms"]["n"], 0)
        self.assertIsNone(s["latency_ms"]["median"])

    def test_indicator_events_are_supplementary(self):
        rec = recording("Y", "S1", duration_s=60)
        frames = make_trace(60, 4, 10, events={31.5: ["YAWN STARTED"]})
        labels = [event("y1", "yawn", 30, 33, "no", "Y"), event("y2", "yawn", 45, 48, "no", "Y")]
        ev = evaluate_recording(rec, labels, RecordingRun("Y", OK, frames=frames), self.cfg, FACE_LOSS_CFG)
        s = aggregate([ev])
        self.assertEqual(s["indicator_recall"]["yawn"]["detected"], 1)
        self.assertEqual(s["indicator_recall"]["yawn"]["events"], 2)
        self.assertEqual(s["false_alarms"], 0)

    def test_breakdowns_by_condition_flag_small_groups(self):
        evs = []
        for rid, sid, glasses in (("A", "S1", "none"), ("B", "S2", "clear"), ("C", "S3", "clear")):
            rec = recording(rid, sid, duration_s=60, glasses=glasses)
            evs.append(evaluate_recording(rec, [], RecordingRun(rid, OK, frames=make_trace(60, 4, 10)),
                                          self.cfg, FACE_LOSS_CFG))
        bd = breakdowns(evs, self.cfg)
        self.assertEqual(set(bd), {"subject_id", "lighting", "glasses", "head_position", "camera_distance"})
        self.assertEqual(bd["glasses"]["clear"]["recordings_evaluated"], 2)
        self.assertTrue(bd["glasses"]["clear"]["insufficient_positive_events"])
        self.assertTrue(bd["glasses"]["none"]["insufficient_non_event_time"])

    def test_dataset_summary_counts(self):
        recs = [recording("A", "S1", glasses="none", duration_s=100),
                recording("B", "S1", glasses="clear", duration_s=50, lighting="dim_indoor"),
                recording("C", "S2", glasses="clear", duration_s=30)]
        labels = {"A": [event("a", "blink", 10, 40, "no", "A"), event("b", "eye_closure", 50, 54, "yes", "A")],
                  "B": [event("c", "eye_closure", 20, 22, "ignore", "B")], "C": []}
        ds = dataset_summary(recs, labels)
        self.assertEqual((ds["subjects"], ds["recordings"], ds["total_duration_s"]), (2, 3, 180.0))
        self.assertEqual(ds["conditions"]["glasses"]["clear"], {"recordings": 2, "subjects": 2, "duration_s": 80.0})
        self.assertEqual(ds["conditions"]["lighting"]["dim_indoor"]["recordings"], 1)
        self.assertEqual(ds["event_counts"]["eye_closure"],
                         {"count": 2, "duration_s": 6.0, "yes": 1, "no": 0, "ignore": 1})
        self.assertEqual(ds["recording_ids"], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
