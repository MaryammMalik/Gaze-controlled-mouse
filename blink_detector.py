import time

# ── Tuned for Areeb's eyes (EAR open = 0.322) ───────────────────
EAR_CLOSED_THRESHOLD = 0.21   # below this = eye closed
EAR_OPEN_THRESHOLD   = 0.25   # above this = eye fully open again

# Timing constants (all in seconds)
MIN_BLINK_DURATION   = 0.04   # eye must be closed at least 40ms  (filters noise)
MAX_BLINK_DURATION   = 0.45   # eye closed longer than this = not a blink (stare/hold)
BLINK_WINDOW         = 0.55   # time window to collect 2nd or 3rd blink
ACTION_COOLDOWN      = 1.0    # after a click action, ignore blinks for 1 second


class BlinkDetector:
    """
    Tracks blink patterns and fires callbacks:
        on_double_blink()  → left click
        on_triple_blink()  → right click

    Usage:
        detector = BlinkDetector(
            on_double_blink=my_left_click_fn,
            on_triple_blink=my_right_click_fn
        )
        # in your frame loop:
        detector.update(avg_ear)
    """

    def __init__(self, on_double_blink=None, on_triple_blink=None):
        self.on_double_blink = on_double_blink or (lambda: None)
        self.on_triple_blink = on_triple_blink or (lambda: None)

        # Internal state
        self._eye_closed      = False
        self._close_start     = 0.0
        self._blink_count     = 0
        self._window_start    = 0.0
        self._last_action_at  = 0.0

        # For the UI overlay
        self.last_action      = "none"    # "double" | "triple" | "none"
        self.last_action_time = 0.0
        self.blink_count_now  = 0         # live count in current window

    def update(self, avg_ear: float):
        """
        Call this every frame with the current average EAR value.
        Returns the current blink count in the active window (0/1/2/3).
        """
        now = time.time()

        # ── Eye just closed ─────────────────────────────────────
        if avg_ear < EAR_CLOSED_THRESHOLD and not self._eye_closed:
            self._eye_closed  = True
            self._close_start = now

        # ── Eye just opened ──────────────────────────────────────
        elif avg_ear > EAR_OPEN_THRESHOLD and self._eye_closed:
            self._eye_closed = False
            closed_duration  = now - self._close_start

            # Only count if it was a real blink (not too short, not too long)
            if MIN_BLINK_DURATION <= closed_duration <= MAX_BLINK_DURATION:

                # Start a new window on the first blink
                if self._blink_count == 0:
                    self._window_start = now

                self._blink_count += 1
                self.blink_count_now = self._blink_count

        # ── Check if window has expired ──────────────────────────
        if self._blink_count > 0:
            window_age = now - self._window_start

            if window_age > BLINK_WINDOW:
                # Window closed — fire action if cooldown allows
                if now - self._last_action_at > ACTION_COOLDOWN:
                    self._fire_action(self._blink_count, now)

                # Reset regardless
                self._blink_count    = 0
                self.blink_count_now = 0

        return self.blink_count_now

    def _fire_action(self, count: int, now: float):
        if count == 2:
            self.last_action      = "double"
            self.last_action_time = now
            self._last_action_at  = now
            self.on_double_blink()

        elif count >= 3:
            self.last_action      = "triple"
            self.last_action_time = now
            self._last_action_at  = now
            self.on_triple_blink()

        # Single blinks are intentionally ignored — normal blinking happens all the time

    def get_status_text(self):
        """Returns a short string for the on-screen overlay."""
        now = time.time()
        if now - self.last_action_time < 0.8:
            if self.last_action == "double":
                return "LEFT CLICK!"
            elif self.last_action == "triple":
                return "RIGHT CLICK!"
        if self.blink_count_now == 1:
            return "1 blink..."
        if self.blink_count_now == 2:
            return "2 blinks..."
        return ""


# ── Quick standalone test ────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import os
    from camera_selector import select_camera
    from eye_detector import process_frame, draw_debug

    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    clicked_log = []

    def on_double():
        msg = f"[{time.strftime('%H:%M:%S')}] DOUBLE BLINK → LEFT CLICK"
        print(msg)
        clicked_log.append(msg)

    def on_triple():
        msg = f"[{time.strftime('%H:%M:%S')}] TRIPLE BLINK → RIGHT CLICK"
        print(msg)
        clicked_log.append(msg)

    detector = BlinkDetector(
        on_double_blink=on_double,
        on_triple_blink=on_triple
    )

    print("\n" + "="*50)
    print("  BLINK DETECTION TEST")
    print("="*50)
    print("Double blink = LEFT CLICK  (printed in terminal)")
    print("Triple blink = RIGHT CLICK (printed in terminal)")
    print("Single blink = ignored (normal blinking)\n")
    print("Press Q to quit.\n")

    result = select_camera()
    if result is None:
        exit()

    cap, _ = result

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        data  = process_frame(frame)
        frame = draw_debug(frame, data)

        if data:
            count = detector.update(data["avg_ear"])
            status = detector.get_status_text()

            # Show blink status on screen
            if status:
                color = (0, 255, 255) if "CLICK" in status else (200, 200, 0)
                cv2.putText(frame, status, (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # Show live blink count
            if count > 0:
                cv2.putText(frame, f"Blinks in window: {count}", (20, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

        cv2.imshow("Blink Test - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("Session summary:")
    if clicked_log:
        for entry in clicked_log:
            print(" ", entry)
    else:
        print("  No clicks triggered this session.")
    print("="*50)
