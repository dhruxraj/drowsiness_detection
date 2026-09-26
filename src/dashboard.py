"""
Real-time dashboard (OpenCV window).

Left : live camera feed with face box, eye & mouth landmarks, warning banners.
Right: measurement panel - state, EAR, MAR, head pose, PERCLOS, blink rate, yawns,
       score bar, alarm status, FPS.
Keys : q/Esc quit | c re-calibrate | t test alarm | r reset statistics
"""
from __future__ import annotations

import cv2
import numpy as np

from .decision import DetectionState
from .metrics import (INNER_LIPS_CONTOUR, LEFT_EYE_CONTOUR, LEFT_EYE_EAR, MOUTH_MAR,
                      RIGHT_EYE_CONTOUR, RIGHT_EYE_EAR)

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, YELLOW, ORANGE, RED = (80, 200, 80), (0, 215, 255), (0, 140, 255), (40, 40, 230)
WHITE, GREY, CYAN, DARK = (240, 240, 240), (150, 150, 150), (230, 200, 0), (30, 30, 30)

STATE_COLORS = {
    DetectionState.ALERT: GREEN,
    DetectionState.BLINKING: CYAN,
    DetectionState.EYES_CLOSED: ORANGE,
    DetectionState.YAWNING: YELLOW,
    DetectionState.HEAD_DROP: ORANGE,
    DetectionState.LOOKING_AWAY: YELLOW,
    DetectionState.DROWSY: RED,
    DetectionState.NO_FACE: GREY,
    DetectionState.CALIBRATING: CYAN,
}


class Dashboard:
    def __init__(self, cfg_display, mirror: bool):
        self.cfg = cfg_display
        self.mirror = mirror
        self.panel_w = int(cfg_display.panel_width)
        self.window = cfg_display.window_name
        self._blink = 0
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    # ------------------------------------------------------------------ drawing helpers
    @staticmethod
    def _text(img, txt, org, scale=0.5, color=WHITE, thick=1):
        cv2.putText(img, txt, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)

    def _draw_face(self, img, pts, closed: bool):
        x0, y0 = pts.min(axis=0).astype(int)
        x1, y1 = pts.max(axis=0).astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), GREEN, 1)
        if self.cfg.draw_mesh_points:
            for p in pts[::3]:
                cv2.circle(img, tuple(p.astype(int)), 1, GREY, -1)
        eye_col = ORANGE if closed else CYAN
        if self.cfg.draw_eyes:
            for contour in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR):
                cv2.polylines(img, [pts[contour].astype(np.int32)], True, eye_col, 1, cv2.LINE_AA)
            for idx in RIGHT_EYE_EAR + LEFT_EYE_EAR:
                cv2.circle(img, tuple(pts[idx].astype(int)), 2, YELLOW, -1)
        if self.cfg.draw_mouth:
            cv2.polylines(img, [pts[INNER_LIPS_CONTOUR].astype(np.int32)], True, GREEN, 1, cv2.LINE_AA)
            for idx in MOUTH_MAR:
                cv2.circle(img, tuple(pts[idx].astype(int)), 2, YELLOW, -1)

    def _banner(self, img, text, color, y):
        w = img.shape[1]
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
        cv2.rectangle(img, (0, y - th - 12), (w, y + 10), color, -1)
        cv2.putText(img, text, ((w - tw) // 2, y), FONT, 0.7, WHITE, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ main render
    def render(self, frame, pts, st, score, decision, fps, thresholds, calib_progress=None):
        self._blink = (self._blink + 1) % 20
        video = frame.copy()
        if pts is not None:
            self._draw_face(video, pts, bool(st and st.eyes_closed))
        if self.mirror:
            video = cv2.flip(video, 1)
        h, w = video.shape[:2]

        # --- overlays on the video
        if decision.alarm_on:
            if self._blink < 10:
                cv2.rectangle(video, (0, 0), (w - 1, h - 1), RED, 12)
            self._banner(video, "DROWSINESS DETECTED - TAKE A BREAK", RED, 45)
        elif decision.driver_not_visible:
            self._banner(video, "DRIVER NOT VISIBLE - CHECK CAMERA", GREY, 45)
        elif decision.fatigue_warning:
            self._banner(video, "FATIGUE SIGNS - PLAN A BREAK SOON", ORANGE, 45)
        if calib_progress is not None:
            self._banner(video, "CALIBRATING: look ahead, eyes open", CYAN, h // 2)
            cv2.rectangle(video, (40, h // 2 + 25), (w - 40, h // 2 + 40), WHITE, 1)
            cv2.rectangle(video, (40, h // 2 + 25),
                          (40 + int((w - 80) * calib_progress), h // 2 + 40), CYAN, -1)
        self._text(video, "q quit | c calibrate | t test alarm | r reset", (10, h - 10), 0.45, GREY)

        # --- side panel
        panel = np.full((h, self.panel_w, 3), DARK, dtype=np.uint8)
        state = decision.state
        col = STATE_COLORS.get(state, WHITE)
        cv2.rectangle(panel, (8, 8), (self.panel_w - 8, 52), col, -1)
        (tw, _), _ = cv2.getTextSize(state.value, FONT, 0.75, 2)
        cv2.putText(panel, state.value, ((self.panel_w - tw) // 2, 39), FONT, 0.75,
                    (0, 0, 0) if col != RED else WHITE, 2, cv2.LINE_AA)

        def fmt(v, spec=".3f"):
            return "--" if v is None else format(v, spec)

        rows = []
        if st is not None:
            ear_col = ORANGE if st.eyes_closed else WHITE
            rows = [
                (f"EAR   {fmt(st.ear)}  (<{thresholds['ear']:.3f})", ear_col),
                (f"MAR   {fmt(st.mar, '.2f')}   (>{thresholds['mar']:.2f})",
                 YELLOW if st.yawning else WHITE),
                (f"Pitch {st.pitch_rel:+5.1f}  Yaw {st.yaw_rel:+5.1f}",
                 ORANGE if st.head_down else YELLOW if st.looking_away else WHITE),
                (f"Eyes closed  {st.closure_duration:4.1f} s", ear_col),
                (f"PERCLOS      {st.perclos * 100:4.1f} %", WHITE),
                (f"Blink rate   {fmt(st.blink_rate, '4.1f')} /min", WHITE),
                (f"Blinks {st.total_blinks}  Yawns {st.total_yawns}  Nods {st.total_nods}", WHITE),
                (f"Face lost    {st.face_lost_duration:4.1f} s",
                 GREY if st.face_lost_duration == 0 else ORANGE),
            ]
        y = 80
        for txt, c in rows:
            self._text(panel, txt, (12, y), 0.48, c)
            y += 26

        # score bar
        y += 6
        total = score.total if score is not None else 0.0
        thr = thresholds["score"]
        self._text(panel, f"Drowsiness score {total:5.1f}", (12, y), 0.5, WHITE)
        y += 10
        bar_w = self.panel_w - 24
        frac = min(total / (thr * 1.2), 1.0)
        bar_col = GREEN if total < thr * 0.5 else YELLOW if total < thr else RED
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 16), GREY, 1)
        cv2.rectangle(panel, (12, y), (12 + int(bar_w * frac), y + 16), bar_col, -1)
        tx = 12 + int(bar_w / 1.2)
        cv2.line(panel, (tx, y - 3), (tx, y + 19), WHITE, 2)
        y += 34
        if score is not None:
            self._text(panel, f"acute {score.acute:5.1f}  history {score.cumulative:4.1f}", (12, y), 0.45, GREY)
        y += 30
        alarm_txt, alarm_col = ("ALARM: ON", RED) if decision.alarm_on else ("ALARM: OFF", GREEN)
        self._text(panel, alarm_txt, (12, y), 0.65, alarm_col, 2)
        self._text(panel, f"FPS {fps:4.1f}", (12, h - 14), 0.5, GREY)

        return np.hstack([video, panel])

    def show(self, img) -> int:
        cv2.imshow(self.window, img)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        cv2.destroyWindow(self.window)
