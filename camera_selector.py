import cv2
import time
import os
import numpy as np
import requests

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# IP_WEBCAM_BASE = "http://192.168.1.4:8080"
#IP_WEBCAM_BASE = "http://192.168.1.6:8080"
IP_WEBCAM_BASE = "http://192.168.0.105:8080"
PHONE_ROTATION = 1  # 90 counter-clockwise — change if still sideways


def rotate_frame(frame, rotation):
    if rotation == 1:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 2:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 3:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class IPWebcamStream:
    """
    Reads frames from IP Webcam using the /shot.jpg endpoint.
    This grabs one JPEG at a time — much more stable than streaming.
    No boundary errors, no freezing.
    """
    def __init__(self, base_url):
        self.url     = f"{base_url}/shot.jpg"
        self.session = requests.Session()
        self.session.timeout = 2.0

    def read(self):
        try:
            resp = self.session.get(self.url, timeout=2.0)
            if resp.status_code == 200:
                arr   = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
        except Exception:
            pass
        return False, None

    def get(self, prop):
        # Fake CAP_PROP values so it behaves like cv2.VideoCapture
        return 0

    def set(self, prop, val):
        pass

    def isOpened(self):
        ret, _ = self.read()
        return ret

    def release(self):
        self.session.close()


def test_ip_webcam(base_url):
    """Test if IP Webcam is reachable and returning frames."""
    try:
        url  = f"{base_url}/shot.jpg"
        resp = requests.get(url, timeout=3.0)
        if resp.status_code == 200:
            arr   = np.frombuffer(resp.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return IPWebcamStream(base_url), frame.shape[1], frame.shape[0]
    except Exception:
        pass
    return None, 0, 0


def scan_local_cameras(max_check=3):
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append((i, cap))
            else:
                cap.release()
        time.sleep(0.05)
    return available


def select_camera():
    print("\n" + "="*50)
    print("   EYE MOUSE — CAMERA SETUP")
    print("="*50)

    options = []

    # Check phone camera
    print("\nChecking IP Webcam (phone camera)...")
    stream, w, h = test_ip_webcam(IP_WEBCAM_BASE)
    if stream is not None:
        options.append(("phone", stream, f"Phone camera {w}x{h} via IP Webcam (RECOMMENDED)"))
        print(f"  [OK] Phone camera connected! Resolution: {w}x{h}")
    else:
        print("  [X] Phone camera not reachable")
        print("      Make sure IP Webcam app is running and on same WiFi")

    # Check laptop camera
    print("\nChecking laptop camera...")
    local_cams = scan_local_cameras()
    added = False
    for idx, cap in local_cams:
        if not added:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            options.append(("local", cap, "Laptop built-in camera (640x480)"))
            added = True
            print(f"  [OK] Laptop camera found (index {idx})")
        else:
            cap.release()

    if not options:
        print("\n[ERROR] No cameras found!")
        input("Press Enter to exit.")
        return None

    print(f"\nFound {len(options)} camera(s):\n")
    for i, (_, _, label) in enumerate(options):
        print(f"  [{i+1}] {label}")
    print()

    if len(options) == 1:
        choice = 0
        print(f"Auto-selecting: {options[0][2]}")
    else:
        while True:
            try:
                raw    = input(f"Enter your choice (1 to {len(options)}): ").strip()
                choice = int(raw) - 1
                if 0 <= choice < len(options):
                    break
                print(f"  Enter a number between 1 and {len(options)}.")
            except ValueError:
                print("  Invalid input.")

    kind, cap, label = options[choice]

    # Release cameras we didn't pick
    for i, (_, c, _) in enumerate(options):
        if i != choice:
            c.release()

    print(f"\n[OK] Camera ready: {label}")
    print("="*50 + "\n")
    return cap, kind


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    result = select_camera()
    if result is None:
        exit()

    cap, kind = result
    rotation  = PHONE_ROTATION

    print("Live feed — R=rotate  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        if kind == "phone":
            frame = rotate_frame(frame, rotation)

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"Rotation: {rotation*90} deg  R=rotate  Q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        display = cv2.resize(frame, (960, 540)) if w > 960 else frame
        cv2.imshow("Camera Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            rotation = (rotation + 1) % 4
            print(f"Rotation: {rotation*90} degrees")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if rotation != PHONE_ROTATION:
        print(f"\nUpdate PHONE_ROTATION = {rotation} in camera_selector.py")
    else:
        print("\nRotation is good!")
    input("Press Enter to close.")
