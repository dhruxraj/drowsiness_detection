"""Core library of the evaluation framework.

Pure-Python modules (no camera, MediaPipe or OpenCV needed):
    intervals, eval_config, dataset, splits, trace, matching, metrics, thresholds, report

Modules that import the production pipeline from ``src/`` lazily (only when a real
recording is processed):
    pipeline, runner
"""
