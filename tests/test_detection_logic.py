"""
Scenario tests for the temporal decision logic (no camera needed).
Each test encodes one requirement, e.g. "a normal blink must NOT trigger the alarm".
"""
import unittest

from tests.helpers import CLOSED, OPEN, Pipeline


class TestFalseAlarmRobustness(unittest.TestCase):
    def test_alert_driver_never_alarms(self):
        p = Pipeline()
        p.run(20)
        self.assertFalse(p.alarm_times)
        self.assertEqual(p.states[-1], "ALERT")

    def test_normal_blinks_do_not_alarm(self):
        p = Pipeline()
        for _ in range(15):                 # 15 blinks of 0.2 s, every 3 s
            p.run(2.8)
            p.run(0.2, ear=CLOSED)
        p.run(2)
        self.assertFalse(p.alarm_times)
        self.assertEqual(p.an.total_blinks, 15)
        self.assertIn("BLINKING", p.states)

    def test_brief_closure_does_not_alarm(self):
        p = Pipeline()
        p.run(3)
        p.run(1.0, ear=CLOSED)              # 1.0 s < 1.8 s threshold
        p.run(3)
        self.assertFalse(p.alarm_times)
        self.assertIn("EYES CLOSED", p.states)

    def test_face_detection_failure_does_not_alarm(self):
        p = Pipeline()
        p.run(3)
        p.run(0.5, face=False)              # short dropout (inside grace period)
        p.run(2)
        p.run(8, face=False)                # long loss
        p.run(3)
        self.assertFalse(p.alarm_times)
        self.assertIn("NO FACE", p.states)

    def test_closed_eyes_then_face_lost_does_not_alarm(self):
        p = Pipeline()
        p.run(3)
        p.run(1.2, ear=CLOSED)
        p.run(3, face=False)                # detection failure must not "complete" the closure
        self.assertFalse(p.alarm_times)

    def test_looking_sideways_does_not_alarm(self):
        p = Pipeline()
        p.run(3)
        p.run(4, ear=0.15, yaw=45)          # head turned: EAR looks low but is unreliable
        p.run(2)
        self.assertFalse(p.alarm_times)
        self.assertIn("LOOKING AWAY", p.states)

    def test_small_head_movements_tolerated(self):
        p = Pipeline()
        for k in range(10):                 # +-10 deg pitch / +-20 deg yaw wobble
            p.run(0.5, pitch=10 if k % 2 else -10, yaw=20 if k % 2 else -20)
        p.run(0.3, pitch=25)                # short dip (road bump) < nod_min_duration
        p.run(2)
        self.assertFalse(p.alarm_times)

    def test_talking_is_not_a_yawn(self):
        p = Pipeline()
        for _ in range(10):
            p.run(0.4, mar=0.7)             # mouth open 0.4 s at a time
            p.run(0.3, mar=0.1)
        self.assertEqual(p.an.total_yawns, 0)


class TestDrowsinessDetection(unittest.TestCase):
    def test_prolonged_closure_triggers_alarm_in_time(self):
        p = Pipeline()
        p.run(3)
        start = p.t
        p.run(2.5, ear=CLOSED)
        self.assertTrue(p.alarm_times)
        delay = p.alarm_times[0] - start
        self.assertGreater(delay, 1.5)
        self.assertLess(delay, 2.1)

    def test_alarm_stops_when_driver_alert(self):
        p = Pipeline()
        p.run(3)
        p.run(2.5, ear=CLOSED)
        self.assertTrue(p.alarm_on)
        p.run(1.0)                          # min_duration (3 s) not yet over
        self.assertTrue(p.alarm_on)
        p.run(3.0)
        self.assertFalse(p.alarm_on)

    def test_alarm_continues_while_eyes_closed(self):
        p = Pipeline()
        p.run(3)
        p.run(8, ear=CLOSED)
        self.assertTrue(p.alarm_on)
        self.assertEqual(len(p.alarm_times), 1)   # no on/off flicker

    def test_yawn_detected_without_alarm(self):
        p = Pipeline()
        p.run(3)
        p.run(3, mar=0.9)
        self.assertEqual(p.an.total_yawns, 1)
        self.assertIn("YAWNING", p.states)
        self.assertFalse(p.alarm_times)

    def test_head_drop_triggers_alarm(self):
        p = Pipeline()
        p.run(3)
        p.run(3, pitch=30)
        self.assertTrue(p.alarm_times)
        self.assertIn("HEAD DROP", p.states)

    def test_combined_indicators_trigger_earlier(self):
        # 1.4 s closure alone -> no alarm
        alone = Pipeline()
        alone.run(3)
        alone.run(1.4, ear=CLOSED)
        self.assertFalse(alone.alarm_times)
        # same closure after three yawns -> alarm (score combines indicators)
        tired = Pipeline()
        for _ in range(3):
            tired.run(3)
            tired.run(2.5, mar=0.9)
        tired.run(3)
        tired.run(1.4, ear=CLOSED)
        self.assertTrue(tired.alarm_times)

    def test_fatigue_history_alone_never_alarms(self):
        p = Pipeline()
        for _ in range(5):                  # many yawns + long blinks, eyes otherwise open
            p.run(2)
            p.run(2.5, mar=0.9)
            p.run(2)
            p.run(0.9, ear=CLOSED)
        p.run(5)
        self.assertFalse(p.alarm_times)
        self.assertTrue(p.last[2].fatigue_warning)


if __name__ == "__main__":
    unittest.main()
