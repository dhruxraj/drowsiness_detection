"""Evaluation runner, production-pipeline adapter and traces (evaluation framework, issue #4).

These tests use FAKE pipeline components and UNIT-TEST FIXTURES (synthetic, not real data)
so they run without a camera, MediaPipe or video files. They check the orchestration:
call order (mirroring main.py), timestamps, timing measurement and failure handling.
The real production modules are exercised by evaluation/scripts/evaluate.py on recordings.
"""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.core.pipeline import (DecisionLogic, DecisionOutput, Measurement, apply_overrides,
                                      assess_measurement)
from evaluation.core.runner import CameraFrameSource, StallError, replay_decisions, run_recording
from evaluation.core.trace import (FAILED_OPEN, FAILED_PROCESSING, NO_FACE, NO_FRAMES, OK, STALLED,
                                   read_trace, write_trace)
from tests.evaluation_fixtures import make_trace


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeSource:
    """Yields (frame, video_timestamp); advancing the clock simulates decoding time."""

    def __init__(self, clock, timestamps, acquire_s=0.004, fail_at=None, stall_at=None):
        self.clock, self.timestamps, self.acquire_s = clock, timestamps, acquire_s
        self.fail_at, self.stall_at, self.closed = fail_at, stall_at, False

    def frames(self):
        for i, ts in enumerate(self.timestamps):
            if i == self.stall_at:
                raise StallError("fixture stall")
            self.clock.now += self.acquire_s
            yield {"idx": i, "face": True}, ts

    def close(self):
        self.closed = True


class FakePipeline:
    """Measurement + decision stand-in; advancing the clock simulates processing time."""

    def __init__(self, clock, process_s=0.02, face=lambda i: True, alarm=lambda t: False, fail_at=None):
        self.clock, self.process_s, self.face, self.alarm, self.fail_at = clock, process_s, face, alarm, fail_at
        self.seen_t, self.closed, self.n = [], False, 0

    def measure(self, frame, t):
        if self.n == self.fail_at:
            raise RuntimeError("fixture failure")
        self.n += 1
        self.seen_t.append(t)
        self.clock.now += self.process_s / 2
        if not self.face(frame["idx"]):
            return Measurement(face_detected=False, invalid_reason="face_not_detected")
        return Measurement(True, 0.3, 0.05, 1.0, 2.0, 0.0, True, "", metrics=("M", t))

    def decide(self, metrics, t):
        self.clock.now += self.process_s / 2
        return DecisionOutput(calibrating=t < 1.0, alarm_on=self.alarm(t), alarm_changed=False,
                              state="ALERT", score=12.5, events=["LONG BLINK (0.80s)"] if t == 2.0 else [],
                              calibration_message="CAL OK" if t == 1.0 else "")

    def close(self):
        self.closed = True


class RunRecordingTests(unittest.TestCase):
    def test_timestamps_timing_and_trace_content(self):
        clock = FakeClock()
        pipe = FakePipeline(clock, process_s=0.02, alarm=lambda t: t >= 2.0)
        src = FakeSource(clock, [50.0, 50.5, 51.0, 51.5, 52.0], acquire_s=0.004)
        run = run_recording("FIXTURE", lambda: pipe, lambda: src, clock=clock)
        self.assertEqual(run.status, OK)
        self.assertEqual(pipe.seen_t, [0.0, 0.5, 1.0, 1.5, 2.0])      # relative to the first frame
        self.assertEqual([f.t_s for f in run.frames], [0.0, 0.5, 1.0, 1.5, 2.0])
        for f in run.frames:
            self.assertAlmostEqual(f.processing_ms, 20.0)              # measure + decide only
            self.assertAlmostEqual(f.acquire_ms, 4.0)                  # decoding measured separately
        self.assertEqual([f.alarm_on for f in run.frames], [False, False, False, False, True])
        self.assertEqual([f.calibrating for f in run.frames], [True, True, False, False, False])
        self.assertEqual(run.frames[4].events, ["LONG BLINK (0.80s)"])
        self.assertEqual(run.calibration_message, "CAL OK")
        self.assertTrue(pipe.closed and src.closed)

    def test_face_never_detected(self):
        clock = FakeClock()
        run = run_recording("FIXTURE", lambda: FakePipeline(clock, face=lambda i: False),
                            lambda: FakeSource(clock, [0.0, 0.1, 0.2]), clock=clock)
        self.assertEqual(run.status, NO_FACE)
        self.assertFalse(any(f.face_detected for f in run.frames))
        self.assertEqual(run.frames[0].invalid_reason, "face_not_detected")

    def test_temporary_face_loss_keeps_status_ok(self):
        clock = FakeClock()
        run = run_recording("FIXTURE", lambda: FakePipeline(clock, face=lambda i: i != 1),
                            lambda: FakeSource(clock, [0.0, 0.1, 0.2]), clock=clock)
        self.assertEqual(run.status, OK)
        self.assertEqual([f.face_detected for f in run.frames], [True, False, True])

    def test_open_failure(self):
        def boom():
            raise OSError("cannot open fixture.mp4")
        run = run_recording("FIXTURE", lambda: FakePipeline(FakeClock()), boom)
        self.assertEqual(run.status, FAILED_OPEN)
        self.assertIn("cannot open", run.error)
        self.assertEqual(run.frames, [])

    def test_no_frames(self):
        clock = FakeClock()
        run = run_recording("FIXTURE", lambda: FakePipeline(clock), lambda: FakeSource(clock, []), clock=clock)
        self.assertEqual(run.status, NO_FRAMES)

    def test_failure_during_processing_keeps_partial_trace(self):
        clock = FakeClock()
        pipe = FakePipeline(clock, fail_at=2)
        src = FakeSource(clock, [0.0, 0.1, 0.2, 0.3])
        run = run_recording("FIXTURE", lambda: pipe, lambda: src, clock=clock)
        self.assertEqual(run.status, FAILED_PROCESSING)
        self.assertEqual(len(run.frames), 2)
        self.assertTrue(pipe.closed and src.closed)

    def test_frame_step_processes_every_nth_frame_on_the_video_clock(self):
        clock = FakeClock()
        pipe = FakePipeline(clock)
        run = run_recording("FIXTURE", lambda: pipe, lambda: FakeSource(clock, [10.0, 10.5, 11.0, 11.5, 12.0]),
                            clock=clock, frame_step=2)
        self.assertEqual(run.frame_step, 2)
        self.assertEqual(pipe.seen_t, [0.0, 1.0, 2.0])            # frames 0, 2, 4; t relative to frame 0
        self.assertEqual(run.meta()["frame_step"], 2)
        with self.assertRaises(ValueError):
            run_recording("FIXTURE", lambda: pipe, lambda: FakeSource(clock, []), frame_step=0)

    def test_stall(self):
        clock = FakeClock()
        run = run_recording("FIXTURE", lambda: FakePipeline(clock),
                            lambda: FakeSource(clock, [0.0, 0.1, 0.2], stall_at=2), clock=clock)
        self.assertEqual(run.status, STALLED)
        self.assertEqual(len(run.frames), 2)


class CameraFrameSourceStallTests(unittest.TestCase):
    def test_stall_detected_after_timeout_without_end_of_file(self):
        clock = FakeClock()

        class NeverEndingEmptyCamera:
            finished = False

            def read(self, timeout):
                clock.now += timeout
                return None, None

        src = object.__new__(CameraFrameSource)   # bypass __init__ (would open src.camera.Camera)
        src.camera, src.read_timeout_s, src.stall_timeout_s, src._clock = NeverEndingEmptyCamera(), 1.0, 3.0, clock
        with self.assertRaises(StallError):
            list(src.frames())

    def test_end_of_file_stops_iteration(self):
        clock = FakeClock()

        class TwoFrameCamera:
            def __init__(self):
                self.items = [("f0", 0.0), (None, None), ("f1", 0.033)]
                self.finished = False

            def read(self, timeout):
                if not self.items:
                    self.finished = True
                    return None, None
                return self.items.pop(0)

        src = object.__new__(CameraFrameSource)
        src.camera, src.read_timeout_s, src.stall_timeout_s, src._clock = TwoFrameCamera(), 1.0, 3.0, clock
        self.assertEqual(list(src.frames()), [("f0", 0.0), ("f1", 0.033)])


# ------------------------------------------------------------------ DecisionLogic (main.py order)

class _Log(list):
    pass


def fake_components(log, calib_frames=2, alarm_score=100.0):
    class Analyzer:
        def __init__(self, cfg):
            log.append("TemporalAnalyzer()")

        def apply_calibration(self, ear, mar, pitch0, yaw0):
            log.append(("apply_calibration", ear, mar, pitch0, yaw0))

        def update(self, metrics, t):
            log.append(("analyzer.update", metrics, t))
            return SimpleNamespace(events=["YAWN STARTED"] if t == 3.0 else [], t=t)

    class Scorer:
        def __init__(self, cfg):
            log.append("DrowsinessScorer()")

        def compute(self, st):
            log.append(("scorer.compute", st.t))
            return SimpleNamespace(total=150.0 if st.t >= 4.0 else 10.0)

    class Engine:
        def __init__(self, cfg):
            log.append("DecisionEngine()")

        def calibrating(self):
            log.append("engine.calibrating")
            return SimpleNamespace(alarm_on=False, state=SimpleNamespace(value="CALIBRATING"))

        def reset(self):
            log.append("engine.reset")

        def update(self, st, score, t):
            log.append(("engine.update", t))
            on = score.total >= alarm_score
            return SimpleNamespace(alarm_on=on, alarm_changed=on, state=SimpleNamespace(value="DROWSY" if on else "ALERT"))

    class Calibrator:
        def __init__(self, cfg):
            log.append("Calibrator()")
            self.n = 0
            self.done = False

        def add(self, metrics, t):
            log.append(("calibrator.add", metrics, t))
            self.n += 1
            self.done = self.n >= calib_frames

        def result(self):
            return SimpleNamespace(ear_threshold=0.22, mar_threshold=0.6, pitch0=1.0, yaw0=-2.0, message="CALIBRATION OK")

    return Analyzer, Scorer, Engine, Calibrator


def cfg_ns(calibration=True):
    return SimpleNamespace(calibration=SimpleNamespace(enabled=calibration))


class DecisionLogicTests(unittest.TestCase):
    def test_call_sequence_mirrors_main_py(self):
        log = _Log()
        logic = DecisionLogic(cfg_ns(), components=fake_components(log))
        outs = [logic.step(("M", t), t) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
        self.assertEqual(log[:4], ["TemporalAnalyzer()", "DrowsinessScorer()", "DecisionEngine()", "Calibrator()"])
        self.assertEqual(log[4:], [
            ("calibrator.add", ("M", 0.0), 0.0), "engine.calibrating",
            ("calibrator.add", ("M", 1.0), 1.0), "engine.calibrating",
            ("apply_calibration", 0.22, 0.6, 1.0, -2.0), "engine.reset",
            ("analyzer.update", ("M", 2.0), 2.0), ("scorer.compute", 2.0), ("engine.update", 2.0),
            ("analyzer.update", ("M", 3.0), 3.0), ("scorer.compute", 3.0), ("engine.update", 3.0),
            ("analyzer.update", ("M", 4.0), 4.0), ("scorer.compute", 4.0), ("engine.update", 4.0),
        ])
        self.assertEqual([o.calibrating for o in outs], [True, True, False, False, False])
        self.assertEqual(outs[1].calibration_message, "CALIBRATION OK")
        self.assertEqual(outs[0].state, "CALIBRATING")
        self.assertEqual(outs[3].events, ["YAWN STARTED"])
        self.assertEqual([o.alarm_on for o in outs], [False, False, False, False, True])
        self.assertEqual(outs[4].score, 150.0)

    def test_calibration_disabled_goes_straight_to_detection(self):
        log = _Log()
        logic = DecisionLogic(cfg_ns(calibration=False), components=fake_components(log))
        logic.step(None, 0.0)                                  # face lost: metrics None as in main.py
        self.assertNotIn("Calibrator()", log)
        self.assertIn(("analyzer.update", None, 0.0), log)

    def test_replay_feeds_recorded_measurements_through_the_logic(self):
        frames = make_trace(duration_s=5, fps=1, calibration_end_s=0, face_missing=[(2.0, 3.0)])
        log = _Log()
        replayed = replay_decisions(frames, cfg_ns(), metrics_factory=lambda *v: v,
                                    logic_factory=lambda cfg: DecisionLogic(cfg, components=fake_components(log)))
        calls = [c for c in log if isinstance(c, tuple) and c[0] in ("calibrator.add", "analyzer.update")]
        self.assertEqual(calls[2], ("analyzer.update", None, 2.0))   # face missing -> None, like main.py
        self.assertEqual(calls[0][1], (0.3, 0.05, 0.0, 0.0, 0.0))
        self.assertEqual([f.alarm_on for f in replayed], [False, False, False, False, True])
        self.assertEqual([f.processing_ms for f in replayed], [f.processing_ms for f in frames])


class MeasurementAndConfigTests(unittest.TestCase):
    def test_assess_measurement(self):
        self.assertEqual(assess_measurement(0.3, 0.1, 1.0, 2.0, True), (True, ""))
        self.assertEqual(assess_measurement(float("nan"), 0.1, 1.0, 2.0, True), (False, "ear_invalid"))
        self.assertEqual(assess_measurement(0.3, None, None, 2.0, True), (False, "mar_invalid;head_pose_unavailable"))
        self.assertEqual(assess_measurement(0.3, 0.1, None, None, False), (True, ""))   # head pose disabled

    def test_apply_overrides(self):
        data = {"eyes": {"closure_duration_threshold": 1.8}, "head": {"pitch_down_threshold": 18}}
        out = apply_overrides(data, {"eyes.closure_duration_threshold": 1.5})
        self.assertEqual(out["eyes"]["closure_duration_threshold"], 1.5)
        self.assertEqual(data["eyes"]["closure_duration_threshold"], 1.8)   # original untouched
        with self.assertRaises(KeyError):
            apply_overrides(data, {"eyes.does_not_exist": 1})

    def test_trace_round_trip_is_exact(self):
        frames = make_trace(duration_s=3, fps=4, calibration_end_s=1, face_missing=[(1.0, 1.5)],
                            invalid=[(2.0, 2.25)], alarms=[(2.5, 3.0)], events={0.5: ["HEAD DROP (pitch +28 deg)"]})
        frames[0].ear = 0.1 + 0.2                      # not exactly representable in short decimal
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.csv"
            write_trace(path, frames)
            back = read_trace(path)
        self.assertEqual(len(back), len(frames))
        self.assertEqual(back[0].ear, 0.1 + 0.2)
        self.assertEqual(back[2].events, ["HEAD DROP (pitch +28 deg)"])
        self.assertIsNone(back[4].ear)
        self.assertTrue(math.isnan(back[8].ear))
        self.assertEqual([f.alarm_on for f in back], [f.alarm_on for f in frames])
        self.assertEqual([f.measurement_valid for f in back], [f.measurement_valid for f in frames])


if __name__ == "__main__":
    unittest.main()
