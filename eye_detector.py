import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe setup ──────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# All 468 face landmarks + iris refinement (gives us 478 total points)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,      # IMPORTANT: enables iris landmarks
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Landmark indices ─────────────────────────────────────────────
# Left eye — EAR (Eye Aspect Ratio) points
LEFT_EYE_TOP      = 159
LEFT_EYE_BOTTOM   = 145
LEFT_EYE_LEFT     = 33
LEFT_EYE_RIGHT    = 133

# Right eye — EAR points
RIGHT_EYE_TOP     = 386
RIGHT_EYE_BOTTOM  = 374
RIGHT_EYE_LEFT    = 362
RIGHT_EYE_RIGHT   = 263

# Iris centers (from refine_landmarks)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# EAR threshold — below this = eye is closed
EAR_THRESHOLD = 0.21


def get_landmark_point(landmarks, index, frame_w, frame_h):
    """Convert a normalised landmark to pixel coordinates."""
    lm = landmarks[index]
    return int(lm.x * frame_w), int(lm.y * frame_h)


def compute_ear(landmarks, top, bottom, left, right, frame_w, frame_h):
    """
    Eye Aspect Ratio = vertical distance / horizontal distance.
    When eye is open  → EAR ≈ 0.25–0.35
    When eye is closed → EAR < 0.21
    """
    pt_top    = np.array(get_landmark_point(landmarks, top,    frame_w, frame_h))
    pt_bottom = np.array(get_landmark_point(landmarks, bottom, frame_w, frame_h))
    pt_left   = np.array(get_landmark_point(landmarks, left,   frame_w, frame_h))
    pt_right  = np.array(get_landmark_point(landmarks, right,  frame_w, frame_h))

    vertical   = np.linalg.norm(pt_top - pt_bottom)
    horizontal = np.linalg.norm(pt_left - pt_right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def process_frame(frame):
    """
    Run MediaPipe on one frame.
    Returns a dict with all the info we need, or None if no face found.

    Dict keys:
        left_ear      — float, left eye aspect ratio
        right_ear     — float, right eye aspect ratio
        avg_ear       — float, average of both eyes
        eye_closed    — bool, True if avg EAR < threshold
        left_iris     — (x, y) pixel coords of left iris center
        right_iris    — (x, y) pixel coords of right iris center
        iris_midpoint — (x, y) midpoint between both irises (used for cursor)
        landmarks     — raw landmark list (for drawing)
        frame_w       — frame width
        frame_h       — frame height
    """
    frame_h, frame_w = frame.shape[:2]

    # MediaPipe needs RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # Compute EAR for both eyes
    left_ear = compute_ear(
        landmarks,
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
        LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
        frame_w, frame_h
    )
    right_ear = compute_ear(
        landmarks,
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
        RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
        frame_w, frame_h
    )
    avg_ear = (left_ear + right_ear) / 2.0

    # Iris positions
    left_iris  = get_landmark_point(landmarks, LEFT_IRIS_CENTER,  frame_w, frame_h)
    right_iris = get_landmark_point(landmarks, RIGHT_IRIS_CENTER, frame_w, frame_h)

    # Midpoint between both irises — this drives the cursor
    iris_midpoint = (
        (left_iris[0] + right_iris[0]) // 2,
        (left_iris[1] + right_iris[1]) // 2
    )

    return {
        "left_ear":      left_ear,
        "right_ear":     right_ear,
        "avg_ear":       avg_ear,
        "eye_closed":    avg_ear < EAR_THRESHOLD,
        "left_iris":     left_iris,
        "right_iris":    right_iris,
        "iris_midpoint": iris_midpoint,
        "landmarks":     landmarks,
        "frame_w":       frame_w,
        "frame_h":       frame_h
    }


def draw_debug(frame, data):
    """
    Draw iris dots, eye outlines, and EAR value on the frame.
    Only used for testing — not needed in the final app.
    """
    if data is None:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    lm  = data["landmarks"]
    fw  = data["frame_w"]
    fh  = data["frame_h"]

    # Draw iris dots
    cv2.circle(frame, data["left_iris"],  6, (0, 255, 0), -1)
    cv2.circle(frame, data["right_iris"], 6, (0, 255, 0), -1)
    cv2.circle(frame, data["iris_midpoint"], 4, (0, 255, 255), -1)

    # Draw eye corner dots
    for idx in [LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
                RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT]:
        pt = get_landmark_point(lm, idx, fw, fh)
        cv2.circle(frame, pt, 3, (255, 100, 0), -1)

    # EAR values on screen
    ear_color = (0, 0, 255) if data["eye_closed"] else (0, 255, 0)
    cv2.putText(frame, f"EAR: {data['avg_ear']:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ear_color, 2)

    status = "CLOSED" if data["eye_closed"] else "OPEN"
    cv2.putText(frame, f"Eye: {status}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ear_color, 2)

    cv2.putText(frame, f"Iris mid: {data['iris_midpoint']}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


# ── Quick test — run this file directly ─────────────────────────
if __name__ == "__main__":
    from camera_selector import select_camera
    import os

    # Suppress obsensor warnings
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

    print("Starting eye detection test...")
    print("Look at the camera. Green dots = iris detected.")
    print("Try slowly closing and opening your eyes.")
    print("Watch the EAR value — it drops when eyes close.\n")
    print("Press Q to quit.\n")

    result = select_camera()
    if result is None:
        exit()

    cap, cam_index = result

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost camera feed.")
            break

        frame = cv2.flip(frame, 1)

        data = process_frame(frame)
        frame = draw_debug(frame, data)

        cv2.imshow("Eye Detection Test - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nTest closed.")
