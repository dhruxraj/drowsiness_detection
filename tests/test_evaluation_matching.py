"""Event-level matching of alarm episodes to ground-truth events (evaluation framework, issue #4).

All data below are UNIT-TEST FIXTURES - synthetic, not real recordings or labels.
"""
import unittest

from evaluation.core.matching import AlarmEpisode, extract_episodes, match_alarms
from tests.evaluation_fixtures import event, make_trace

PRE, POST = 0.5, 1.0


def ep(onset, offset=None):
    return AlarmEpisode(onset, offset)


class ExtractEpisodesTests(unittest.TestCase):
    def test_on_off_transitions(self):
        frames = make_trace(duration_s=20, fps=4, calibration_end_s=0, alarms=[(2.0, 5.0), (10.0, 12.5)])
        self.assertEqual(extract_episodes(frames), [ep(2.0, 5.0), ep(10.0, 12.5)])

    def test_alarm_still_on_at_end_of_recording(self):
        frames = make_trace(duration_s=10, fps=4, calibration_end_s=0, alarms=[(8.0, 99.0)])
        self.assertEqual(extract_episodes(frames), [ep(8.0, None)])

    def test_single_frame_alarm_is_one_episode(self):
        frames = make_trace(duration_s=5, fps=4, calibration_end_s=0, alarms=[(1.0, 1.25)])
        self.assertEqual(extract_episodes(frames), [ep(1.0, 1.25)])

    def test_no_alarm(self):
        self.assertEqual(extract_episodes(make_trace(duration_s=5, fps=4)), [])


class MatchTests(unittest.TestCase):
    def match(self, events, episodes, eval_start=10.0):
        return match_alarms(events, episodes, eval_start, PRE, POST)

    def test_onset_inside_event_is_a_detection_with_latency(self):
        e = event("e1", "eye_closure", 20.0, 24.0, "yes")
        r = self.match([e], [ep(21.8, 25.0)])
        self.assertEqual(len(r.detections), 1)
        self.assertAlmostEqual(r.detections[0].latency_ms, 1800.0)
        self.assertAlmostEqual(r.detections[0].release_delay_ms, 1000.0)
        self.assertEqual((r.missed, r.false_alarms), ([], []))

    def test_tolerance_window_boundaries_are_inclusive(self):
        e = event("e1", "eye_closure", 20.0, 24.0, "yes")
        early = self.match([e], [ep(19.5, 22.0)])            # start - pre_tolerance
        self.assertEqual(len(early.detections), 1)
        self.assertAlmostEqual(early.detections[0].latency_ms, -500.0)   # negative latency is kept
        late = self.match([e], [ep(25.0, 27.0)])             # end + post_tolerance
        self.assertEqual(len(late.detections), 1)

    def test_onset_outside_window_is_false_alarm_and_event_missed(self):
        e = event("e1", "eye_closure", 20.0, 24.0, "yes")
        r = self.match([e], [ep(19.4, 20.5), ep(25.1, 27.0)])
        self.assertEqual(len(r.detections), 0)
        self.assertEqual(r.missed, [e])
        self.assertEqual(r.false_alarms, [ep(19.4, 20.5), ep(25.1, 27.0)])

    def test_alarm_already_on_before_window_does_not_count_as_detection(self):
        # conservative onset-based rule, documented in evaluation/README.md
        e = event("e1", "eye_closure", 20.0, 24.0, "yes")
        r = self.match([e], [ep(15.0, None)])
        self.assertEqual((len(r.false_alarms), r.missed), (1, [e]))

    def test_second_onset_for_same_event_is_duplicate_not_false_alarm(self):
        e = event("e1", "head_drop", 20.0, 24.0, "yes")
        r = self.match([e], [ep(21.0, 22.0), ep(23.0, 26.0)])
        self.assertEqual(len(r.detections), 1)
        self.assertEqual(r.duplicates, [ep(23.0, 26.0)])
        self.assertEqual(r.false_alarms, [])

    def test_one_onset_detects_at_most_one_event(self):
        a = event("a", "eye_closure", 20.0, 24.0, "yes")
        b = event("b", "eye_closure", 24.5, 28.0, "yes")    # windows overlap around 24-25
        r = self.match([a, b], [ep(24.2, 30.0)])
        self.assertEqual([d.event.event_id for d in r.detections], ["a"])   # earliest eligible event
        self.assertEqual([m.event_id for m in r.missed], ["b"])
        r2 = self.match([a, b], [ep(24.2, 24.4), ep(24.6, 30.0)])
        self.assertEqual(sorted(d.event.event_id for d in r2.detections), ["a", "b"])

    def test_onset_in_ignore_window_is_ignored(self):
        amb = event("i1", "eye_closure", 40.0, 42.0, "ignore")
        r = self.match([amb], [ep(42.9, 45.0)])
        self.assertEqual(r.ignored, [ep(42.9, 45.0)])
        self.assertEqual((r.false_alarms, r.missed), ([], []))

    def test_alarm_during_non_alarm_event_is_false_alarm(self):
        yawn = event("y1", "yawn", 40.0, 43.0, "no")
        blink = event("b1", "blink", 50.0, 80.0, "no")
        r = self.match([yawn, blink], [ep(41.0, 44.0), ep(60.0, 63.0)])
        self.assertEqual(len(r.false_alarms), 2)

    def test_onset_before_evaluation_window_counted_separately(self):
        r = self.match([], [ep(3.0, 6.0), ep(12.0, 15.0)], eval_start=10.0)
        self.assertEqual(r.before_window, [ep(3.0, 6.0)])
        self.assertEqual(r.false_alarms, [ep(12.0, 15.0)])

    def test_configurable_tolerance(self):
        e = event("e1", "eye_closure", 20.0, 24.0, "yes")
        strict = match_alarms([e], [ep(24.5, 26.0)], 0.0, 0.0, 0.0)
        loose = match_alarms([e], [ep(24.5, 26.0)], 0.0, 0.0, 1.0)
        self.assertEqual((len(strict.detections), len(loose.detections)), (0, 1))


if __name__ == "__main__":
    unittest.main()
