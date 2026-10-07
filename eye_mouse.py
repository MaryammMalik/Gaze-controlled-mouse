import cv2
import numpy as np
import pyautogui
import os
import time
import math
from camera_selector import select_camera
from eye_detector import process_frame
from blink_detector import BlinkDetector

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ── ANTI-JITTER: BACK TO A SIMPLE BASELINE (v6) ─────────────────────
# Handover doc Section 8, Attempts 1-6 all failed in different ways
# (Y_GAIN drift, head/iris lag mismatch, One Euro on the wrong side of
# the amplification). Root cause identified in 8.4/8.7: head_center is
# recomputed fresh every frame with NO smoothing, and the vertical
# calibration slope (ay) is large — so tiny per-frame noise in
# head_center gets amplified to full screen-pixel scale BEFORE any
# filter ever sees it. Speed-based filters (EMA, One Euro) can't tell
# amplified noise apart from real movement once they're that big.
#
# Per Section 8.9 step 2: before trying anything adaptive again,
# revert to ONE simple, well-understood baseline and confirm it is at
# least STABLE (even if slow/laggy) before layering complexity back
# in. This is that baseline:
#
#   - No gain multipliers (X_GAIN/Y_GAIN — ruled out in 8.3-8.4)
#   - No separate head/iris EMA matching (ruled out in 8.5-8.6)
#   - No One Euro Filter (ruled out on this side of the amplification, 8.7)
#   - Just ONE single-stage EMA, applied to the final calibrated
#     screen x/y, at a conservative fixed alpha.
#
# Pipeline: raw rel_iris -> linear calibration map -> single EMA
#           (per axis, on screen pixels) -> deadzone -> moveTo
#
# EMA_ALPHA: how much of each new reading to accept per frame.
#   Lower = smoother/steadier but laggier. 0.05 is deliberately
#   conservative — the goal right now is proving stability, not speed.
# v7: filter moved to PRE-calibration rel_iris (raw eye-space), instead
# of post-calibration screen coords. Noise gets smoothed while it's
# still small, before ax/ay blows it up — so the filter can actually
# tell noise apart from real movement. Re-tuned alpha for this much
# smaller coordinate scale (rel_iris is in pixels within the face
# crop, not full-screen pixels).
EMA_ALPHA   = 0.15
DEADZONE_PX = 10       # small safety-net deadzone AFTER calibration


class SimpleEMAFilter:
    """
    Fixed-rate exponential moving average — the plainest possible
    smoothing. No velocity estimation, no adaptive behavior, nothing
    that can itself introduce a new failure mode. Used here purely as
    a stability baseline (see note above) before considering anything
    smarter.
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def filter(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

# ── Gaze-controlled QUIT button ────────────────────────────────────
# A small always-on-top box in the top-right corner. Hold your gaze
# on it for QUIT_DWELL_SECONDS and the app closes itself — no keyboard
# or hand movement needed. Looking away before it fills resets the
# timer, so passing over it briefly won't trigger it by accident.
QUIT_BTN_W          = 160
QUIT_BTN_H          = 100
QUIT_BTN_MARGIN     = 20
QUIT_BTN_X          = SCREEN_W - QUIT_BTN_W - QUIT_BTN_MARGIN
QUIT_BTN_Y          = QUIT_BTN_MARGIN
QUIT_DWELL_SECONDS  = 2.5

# NOTE: SENSITIVITY_X/Y removed — the per-axis linear mapping now fits its
# own scale factor (ax, ay) directly from the 4-corner calibration, so a
# separate hand-tuned sensitivity multiplier is redundant and was actually
# fighting the fit (it scaled eye-space before calibration ever saw it).
# If corner-to-corner range feels too small/large, widen or narrow the
# CORNERS percentages in CalibrationState instead.

EAR_FREEZE_THRESHOLD = 0.23

# ── Landmark averaging ────────────────────────────────────────────
# Instead of using ONE iris landmark point (which is noisy), we average
# MULTIPLE landmarks around each iris to get a much more stable center.
# MediaPipe gives us 5 points per iris when refine_landmarks=True:
# the center point plus 4 boundary points around it.
LEFT_IRIS_POINTS  = [468, 469, 470, 471, 472]
RIGHT_IRIS_POINTS = [473, 474, 475, 476, 477]

NOSE_TIP    = 4
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454


def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def get_iris_center_averaged(landmarks, point_indices, w, h):
    """Average multiple iris boundary points for a much more stable center
    than using a single landmark point."""
    pts = [get_pt(landmarks, idx, w, h) for idx in point_indices]
    return np.mean(pts, axis=0)


def get_head_center(landmarks, w, h):
    nose   = get_pt(landmarks, NOSE_TIP,    w, h)
    lcheek = get_pt(landmarks, LEFT_CHEEK,  w, h)
    rcheek = get_pt(landmarks, RIGHT_CHEEK, w, h)
    return (nose + lcheek + rcheek) / 3.0


def get_iris_relative_to_head(landmarks, w, h):
    try:
        head_center = get_head_center(landmarks, w, h)
        left_iris   = get_iris_center_averaged(landmarks, LEFT_IRIS_POINTS,  w, h)
        right_iris  = get_iris_center_averaged(landmarks, RIGHT_IRIS_POINTS, w, h)
        iris_avg    = (left_iris + right_iris) / 2.0
        relative    = iris_avg - head_center
        return relative, iris_avg, head_center
    except Exception:
        return None, None, None


class CalibrationState:
    """
    Same 4-corner UI flow as before, but fits a SIMPLE INDEPENDENT
    LINEAR mapping per axis instead of a full affine transform:

        screen_x = ax * relx + bx
        screen_y = ay * rely + by

    This is the "direct mapping" idea (screen = scale * eyespace)
    applied per-axis. It's less prone to amplifying noise than a full
    2x2 affine matrix because it can't mix x-noise into y-movement.
    """
    CORNERS = [
        ("TOP-LEFT",     (0.05, 0.05)),
        ("TOP-RIGHT",    (0.95, 0.05)),
        ("BOTTOM-RIGHT", (0.95, 0.95)),
        ("BOTTOM-LEFT",  (0.05, 0.95)),
    ]

    def __init__(self):
        self.step           = 0
        self.done           = False
        self.samples        = []
        self.eye_points     = []
        self.scr_points     = []
        self.ax = self.bx = self.ay = self.by = None
        self._collecting    = False
        self._collect_start = 0.0
        self.COLLECT_TIME   = 2.0

    def current_corner_name(self):
        if self.done or self.step >= len(self.CORNERS):
            return ""
        return self.CORNERS[self.step][0]

    def current_screen_target(self):
        if self.done or self.step >= len(self.CORNERS):
            return None
        rx, ry = self.CORNERS[self.step][1]
        return int(rx * SCREEN_W), int(ry * SCREEN_H)

    def start_collecting(self):
        self._collecting    = True
        self._collect_start = time.time()
        self.samples        = []

    def update(self, rel_iris):
        if self.done or not self._collecting:
            return False
        self.samples.append(rel_iris.copy())
        if time.time() - self._collect_start >= self.COLLECT_TIME and len(self.samples) >= 15:
            # Use MEDIAN instead of mean — robust against outlier jitter frames
            avg = np.median(np.array(self.samples), axis=0)
            scr = self.current_screen_target()
            self.eye_points.append(avg)
            self.scr_points.append(list(scr))
            self._collecting = False
            self.step += 1
            if self.step >= len(self.CORNERS):
                self._build_mapping()
                self.done = True
            return True
        return False

    def _build_mapping(self):
        try:
            eye = np.array(self.eye_points, dtype=np.float64)  # Nx2 (relx, rely)
            scr = np.array(self.scr_points, dtype=np.float64)  # Nx2 (screen_x, screen_y)

            # Independent least-squares line fit per axis: screen = a*eye + b
            self.ax, self.bx = np.polyfit(eye[:, 0], scr[:, 0], 1)
            self.ay, self.by = np.polyfit(eye[:, 1], scr[:, 1], 1)
            print("\n[CAL] Calibration complete! (linear per-axis mapping)")
        except Exception as e:
            print(f"[CAL] ERROR: {e}")
            self.ax = self.bx = self.ay = self.by = None

    def iris_to_screen(self, rel_iris):
        if self.ax is None:
            return None
        try:
            x = self.ax * rel_iris[0] + self.bx
            y = self.ay * rel_iris[1] + self.by
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                return None
            x = float(np.clip(x, 0, SCREEN_W - 1))
            y = float(np.clip(y, 0, SCREEN_H - 1))
            return x, y
        except Exception:
            return None


class EyeMouse:
    def __init__(self):
        self.cal             = CalibrationState()
        cx, cy                = pyautogui.position()
        self._mx              = float(cx)
        self._my              = float(cy)

        # Simple EMA filters — one per screen axis. Applied AFTER the
        # calibration mapping, on final screen pixel coordinates. See
        # the note near the top of this file for why we're back to a
        # plain fixed-rate filter rather than the One Euro Filter.
        self._filter_x = SimpleEMAFilter(EMA_ALPHA)
        self._filter_y = SimpleEMAFilter(EMA_ALPHA)

        self._click_text      = ""
        self._click_time      = 0.0
        self._cal_phase       = "waiting"
        self._phase_start     = time.time()
        self._countdown       = 3
        self._cursor_frozen   = False

        # Gaze-controlled quit button state
        self._quit_hover_start = None
        self.quit_requested    = False

        self.blink_detector = BlinkDetector(
            on_double_blink=self._left_click,
            on_triple_blink=self._right_click
        )

    def _left_click(self):
        try:
            pyautogui.click()
            self._click_text = "LEFT CLICK"
            self._click_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] LEFT CLICK  @ {pyautogui.position()}")
        except Exception as e:
            print(f"[CLICK] left click failed: {e}")

    def _right_click(self):
        try:
            pyautogui.rightClick()
            self._click_text = "RIGHT CLICK"
            self._click_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] RIGHT CLICK @ {pyautogui.position()}")
        except Exception as e:
            print(f"[CLICK] right click failed: {e}")

    def _smooth_move(self, tx, ty):
        if self._cursor_frozen:
            return

        # tx, ty are already-calibrated screen coords, coming from a
        # rel_iris that was smoothed BEFORE calibration (see update()).
        # Deadzone is the only thing happening here now.
        if abs(tx - self._mx) > DEADZONE_PX or abs(ty - self._my) > DEADZONE_PX:
            self._mx = tx
            self._my = ty
            try:
                pyautogui.moveTo(int(self._mx), int(self._my))
            except Exception:
                pass

    def update(self, data):
        if data is None:
            self._cursor_frozen = False
            return

        lm  = data["landmarks"]
        fw  = data["frame_w"]
        fh  = data["frame_h"]
        ear = data["avg_ear"]

        if ear < EAR_FREEZE_THRESHOLD:
            if not self._cursor_frozen:
                self._cursor_frozen = True
                # Reset the filters so blink-era frames (where the
                # iris landmark is distorted by the closing eyelid)
                # don't pull the cursor when we unfreeze
                self._filter_x.reset()
                self._filter_y.reset()
        else:
            self._cursor_frozen = False

        rel_iris, iris_abs, head_center = get_iris_relative_to_head(lm, fw, fh)
        if rel_iris is None:
            return

        if not self.cal.done:
            now = time.time()
            if self._cal_phase == "waiting":
                if now - self._phase_start >= self._countdown:
                    self.cal.start_collecting()
                    self._cal_phase = "collecting"
            elif self._cal_phase == "collecting":
                # Median-of-samples per corner already rejects outliers,
                # so raw per-frame rel_iris is fine to feed calibration.
                corner_done = self.cal.update(rel_iris * 1.0)
                if corner_done:
                    if self.cal.done:
                        self._cal_phase = "done"
                    else:
                        self._cal_phase   = "next_corner"
                        self._phase_start = now
            elif self._cal_phase == "next_corner":
                if now - self._phase_start >= 2.0:
                    self.cal.start_collecting()
                    self._cal_phase = "collecting"
            return

        # Smooth rel_iris HERE, before it hits the ax/ay calibration
        # multiplication — this is the pre-calibration fix (v7). Noise
        # gets damped while still small; only the clean signal gets
        # scaled up to screen pixels afterward.
        smoothed_relx = self._filter_x.filter(rel_iris[0])
        smoothed_rely = self._filter_y.filter(rel_iris[1])
        smoothed_rel  = np.array([smoothed_relx, smoothed_rely])

        result = self.cal.iris_to_screen(smoothed_rel)
        if result:
            self._smooth_move(result[0], result[1])
        self.blink_detector.update(ear)
        self._check_quit_button()

    def _check_quit_button(self):
        """Check if the actual cursor position is resting on the QUIT
        button, and how long it's been there. Sets self.quit_requested
        once the dwell time is reached."""
        mx, my = self._mx, self._my
        inside = (QUIT_BTN_X <= mx <= QUIT_BTN_X + QUIT_BTN_W and
                  QUIT_BTN_Y <= my <= QUIT_BTN_Y + QUIT_BTN_H)

        now = time.time()
        if inside:
            if self._quit_hover_start is None:
                self._quit_hover_start = now
            elif now - self._quit_hover_start >= QUIT_DWELL_SECONDS:
                self.quit_requested = True
        else:
            self._quit_hover_start = None

    def quit_button_progress(self):
        """Returns 0.0-1.0 fill fraction for drawing the button, or 0 if not hovering."""
        if self._quit_hover_start is None:
            return 0.0
        return min((time.time() - self._quit_hover_start) / QUIT_DWELL_SECONDS, 1.0)


    def draw_overlay(self, frame, data):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 140), (10, 10, 10), -1)

        if data is None:
            cv2.putText(frame, "No face — move closer",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            return frame

        lm = data["landmarks"]
        fw = data["frame_w"]
        fh = data["frame_h"]
        rel_iris, iris_abs, head_center = get_iris_relative_to_head(lm, fw, fh)

        if iris_abs is not None:
            dot_color = (0, 100, 255) if self._cursor_frozen else (0, 255, 80)
            cv2.circle(frame, tuple(iris_abs.astype(int)),    6, dot_color,     -1)
            cv2.circle(frame, tuple(head_center.astype(int)), 4, (255, 100, 0), -1)

        if not self.cal.done:
            now    = time.time()
            corner = self.cal.current_corner_name()

            if self._cal_phase == "waiting":
                remaining = max(0, self._countdown - (now - self._phase_start))
                cv2.putText(frame, f"Look at {corner} corner of screen",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, f"Starting in {remaining:.1f}s ...",
                            (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
            elif self._cal_phase == "collecting":
                pct = int(min((time.time() - self.cal._collect_start) / self.cal.COLLECT_TIME * 100, 100))
                cv2.putText(frame, f"HOLD GAZE — {corner}  ({pct}%)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
                bar_w = int((w - 20) * pct / 100)
                cv2.rectangle(frame, (10, 50), (w - 10, 68), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 50), (10 + bar_w, 68), (0, 200, 80), -1)
            elif self._cal_phase == "next_corner":
                nxt = self.cal.CORNERS[self.cal.step][0] if self.cal.step < len(self.cal.CORNERS) else "done"
                cv2.putText(frame, f"Done! Now look at: {nxt}",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Corner {self.cal.step + 1} / {len(self.cal.CORNERS)}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
            return frame

        ear_color = (0, 80, 255) if data["eye_closed"] else (0, 220, 80)
        cv2.putText(frame, f"EAR: {data['avg_ear']:.3f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        if self._cursor_frozen:
            cv2.putText(frame, "FROZEN", (w - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        mx, my = pyautogui.position()
        cv2.putText(frame, f"Cursor: ({mx}, {my})  |  R=recalibrate",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        bstatus = self.blink_detector.get_status_text()
        if bstatus:
            color = (0, 255, 255) if "CLICK" in bstatus else (160, 160, 255)
            cv2.putText(frame, bstatus, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if time.time() - self._click_time < 1.0:
            cv2.putText(frame, self._click_text, (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.putText(frame, "Double blink=Left  Triple blink=Right  R=Recal  Q=Quit  |  Gaze QUIT box=Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        return frame


def make_quit_button_image(progress):
    """Renders the QUIT button as an image: a red border box that fills
    green from left to right as dwell progress increases."""
    img = np.full((QUIT_BTN_H, QUIT_BTN_W, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (QUIT_BTN_W - 1, QUIT_BTN_H - 1), (0, 0, 255), 3)

    fill_w = int((QUIT_BTN_W - 10) * progress)
    if fill_w > 0:
        cv2.rectangle(img, (5, QUIT_BTN_H - 20), (5 + fill_w, QUIT_BTN_H - 10), (0, 220, 80), -1)

    cv2.putText(img, "QUIT", (QUIT_BTN_W // 2 - 45, QUIT_BTN_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(img, "Look here to exit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    return img


def main():
    print("\n" + "="*55)
    print("   EYE MOUSE v6 — by Areeb (Simple-EMA Stability Baseline)")
    print("="*55)
    print("Look at each RED DOT — hold gaze for 2 seconds.")
    print("4 corners: TOP-LEFT -> TOP-RIGHT -> BOTTOM-RIGHT -> BOTTOM-LEFT")
    print()
    print("Double blink=Left click   Triple blink=Right click")
    print(f"Look at the QUIT box (top-right corner) for {QUIT_DWELL_SECONDS}s to exit.")
    print("R=Recalibrate   Q=Quit (keyboard also still works)")
    print("="*55 + "\n")

    result = select_camera()
    if result is None:
        return

    cap, kind = result
    app       = EyeMouse()

    cv2.namedWindow("QUIT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QUIT", QUIT_BTN_W, QUIT_BTN_H)
    cv2.moveWindow("QUIT", QUIT_BTN_X, QUIT_BTN_Y)

    cal_screen    = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    prev_cal_step = -1

    cv2.namedWindow("cal_dot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("cal_dot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cam_win = "Eye Mouse v6 — Q to quit"
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cam_win, 480, 360)
    cv2.moveWindow(cam_win, SCREEN_W - 500, SCREEN_H - 420)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Only flip for laptop camera — phone camera via IP webcam
        # is already correctly oriented after rotation in camera_selector
        if kind == "local":
            frame = cv2.flip(frame, 1)

        try:
            data  = process_frame(frame)
            app.update(data)
            frame = app.draw_overlay(frame, data)
        except Exception as e:
            print(f"[FRAME] Error: {e}")
            continue

        if not app.cal.done:
            target = app.cal.current_screen_target()
            if target and app.cal.step != prev_cal_step:
                cal_screen[:] = 15
                cv2.circle(cal_screen, target, 30, (0, 0, 255), -1)
                cv2.circle(cal_screen, target, 34, (255, 255, 255), 3)
                corner_name = app.cal.current_corner_name()
                lx = 50 if target[0] < SCREEN_W // 2 else SCREEN_W - 350
                ly = 80 if target[1] < SCREEN_H // 2 else SCREEN_H - 50
                cv2.putText(cal_screen, f"Look at this dot: {corner_name}",
                            (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                prev_cal_step = app.cal.step
            cv2.imshow("cal_dot", cal_screen)
            # Bring calibration window to front every frame during calibration
            cv2.setWindowProperty("cal_dot", cv2.WND_PROP_TOPMOST, 1)
        else:
            try:
                cv2.destroyWindow("cal_dot")
                cv2.resizeWindow(cam_win, 480, 360)
                cv2.moveWindow(cam_win, 20, 20)
            except Exception:
                pass

        cv2.imshow(cam_win, frame)

        # ── Gaze QUIT button ──────────────────────────────────────
        # Only active once calibration is done — otherwise your gaze
        # is busy on the calibration dots and shouldn't trigger it.
        if app.cal.done:
            progress = app.quit_button_progress()
            cv2.imshow("QUIT", make_quit_button_image(progress))
            cv2.setWindowProperty("QUIT", cv2.WND_PROP_TOPMOST, 1)
            if app.quit_requested:
                print("\n[QUIT] Gaze quit button triggered.")
                break
        else:
            cv2.imshow("QUIT", make_quit_button_image(0.0))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            app.cal           = CalibrationState()
            app._cal_phase    = "waiting"
            app._phase_start  = time.time()
            app._filter_x.reset()
            app._filter_y.reset()
            app._quit_hover_start = None
            app.quit_requested    = False
            prev_cal_step     = -1
            cal_screen[:]     = 15
            cv2.namedWindow("cal_dot", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("cal_dot", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow(cam_win, 480, 360)
            cv2.moveWindow(cam_win, SCREEN_W - 500, SCREEN_H - 420)
            print("\n[RECAL] Recalibrating...")

    cap.release()
    cv2.destroyAllWindows()
    print("\nEye Mouse closed.")


if __name__ == "__main__":
    main()