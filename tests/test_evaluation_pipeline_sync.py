"""Guard against drift between main.py and the evaluation adapter (evaluation framework, issue #4).

The evaluator must exercise the real production pipeline. ``evaluation/core/pipeline.py``
composes the same ``src`` modules in the same order as the per-frame loop in ``main.py``
(without dashboard, alarm hardware and logger). If one of the mirrored calls disappears
from ``main.py``, this test fails so the adapter is updated in the same change.
"""
import ast
import re
import unittest
from pathlib import Path

from evaluation.core.pipeline import MIRRORED_MAIN_CALLS

ROOT = Path(__file__).resolve().parents[1]


class PipelineSyncTests(unittest.TestCase):
    def setUp(self):
        self.main_py = (ROOT / "main.py").read_text(encoding="utf-8")
        self.adapter = (ROOT / "evaluation" / "core" / "pipeline.py").read_text(encoding="utf-8")

    def test_mirrored_calls_still_exist_in_main_py(self):
        missing = [p for p in MIRRORED_MAIN_CALLS if not re.search(p, self.main_py)]
        self.assertEqual(missing, [], "main.py changed - update evaluation/core/pipeline.py to mirror "
                                      f"the per-frame loop, then update MIRRORED_MAIN_CALLS: {missing}")

    def test_stages_appear_in_the_same_order_in_main_py(self):
        stages = (r"\.process\(frame\)", r"detector\.detect\(", r"head\.estimate\(",
                  r"calibrator\.add\(", r"engine\.calibrating\(\)", r"analyzer\.apply_calibration\(",
                  r"analyzer\.update\(", r"scorer\.compute\(", r"engine\.update\(")
        found = [re.search(p, self.main_py) for p in stages]
        self.assertTrue(all(found), "a pipeline stage is missing from main.py")
        pos = [m.start() for m in found]
        self.assertEqual(pos, sorted(pos), "per-frame stage order in main.py differs from the adapter")
        # every stage is also called by the adapter (its call order is checked with fake
        # components in test_evaluation_runner.DecisionLogicTests)
        self.assertTrue(all(re.search(p, self.adapter) for p in stages))

    def test_adapter_uses_the_production_modules(self):
        tree = ast.parse(self.adapter)
        imported = {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    and n.module and n.module.startswith("src.") for a in n.names}
        for name in ("src.preprocessing.Preprocessor", "src.landmark_detector.FaceLandmarkDetector",
                     "src.head_pose.HeadPoseEstimator", "src.metrics.compute_ear", "src.metrics.compute_mar",
                     "src.temporal_analyzer.FrameMetrics", "src.temporal_analyzer.TemporalAnalyzer",
                     "src.drowsiness_scorer.DrowsinessScorer", "src.decision.DecisionEngine",
                     "src.calibration.Calibrator", "src.config.load_config"):
            self.assertIn(name, imported)
        runner = (ROOT / "evaluation" / "core" / "runner.py").read_text(encoding="utf-8")
        self.assertIn("from src.camera import Camera", runner)

    def test_evaluation_code_contains_no_own_detection_thresholds(self):
        # detection thresholds must come from config.yaml, never be re-implemented here
        for name in ("pipeline.py", "runner.py"):
            text = (ROOT / "evaluation" / "core" / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"ear\s*[<>]=?\s*0\.\d")
            self.assertNotRegex(text, r"mar\s*[<>]=?\s*0\.\d")


if __name__ == "__main__":
    unittest.main()
