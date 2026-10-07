# 👁️ Gaze-Controlled Mouse for Disabled People

A hands-free mouse controlled entirely by eye movements and blinks — no keyboard or mouse required. Built using a standard laptop webcam and free, open-source tools, for people with motor disabilities who cannot use conventional input devices.

- **Look** in any direction → cursor moves there
- **Double blink** → Left click
- **Triple blink** → Right click

---

## Why This Project

Most assistive input devices are expensive or require specialized hardware. This project shows that a functional, hands-free input system can be built with nothing more than a regular webcam and open-source libraries — making accessibility tech more reachable.

## How It Works

The app captures live video from the webcam and uses **MediaPipe Face Mesh** to detect facial landmarks and locate the iris in each eye. The iris position determines where on screen the user is looking, and the cursor is moved there in real time.

Before use, a quick **4-point calibration** runs: a red dot appears at each corner of the screen, and the user holds their gaze on it for 1.5 seconds. This builds a mapping between eye position and screen coordinates for that specific user and camera angle.

Clicks are triggered using the **Eye Aspect Ratio (EAR)** — a measurement of how open or closed the eye is:
- EAR ≈ 0.28–0.35 → eye open
- EAR < 0.21 → eye closed (blink)

To avoid accidental clicks from normal blinking, the system only acts on **deliberate multi-blink gestures**:
- After the first blink, a 0.55s window opens for a second blink (→ left click)
- A third blink within that window triggers a right click instead
- A 1-second cooldown follows every action

**Head movement compensation:** Moving your head shifts the iris position in the camera frame even without an actual eye movement. To correct for this, the system tracks the face center using nose/cheek landmarks each frame, and measures iris position *relative to the face center* — so only real eye movement drives the cursor.

## Project Structure

| File | Purpose |
|---|---|
| `camera_selector.py` | Detects available cameras and lets the user choose one (laptop cam or DroidCam) |
| `eye_detector.py` | Runs MediaPipe on each frame; returns iris position and EAR value |
| `blink_detector.py` | Detects double/triple blink patterns and fires click actions |
| `eye_mouse.py` | Main file — runs calibration, maps eye position to cursor, handles the app loop |

## Tools & Libraries

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10.11 | Core language (must be 3.10 — MediaPipe doesn't support 3.11+) |
| MediaPipe | 0.10.9 | Face & iris landmark detection |
| OpenCV | 4.8.1.78 | Camera capture, frame processing, on-screen overlay |
| NumPy | 1.26.4 | Calibration math and signal smoothing |
| PyAutoGUI | 0.9.54 | Cursor movement and click execution |
| Pillow | 10.2.0 | Image support (required by PyAutoGUI) |

## Installation

```bash
py -3.10 -m venv eye_mouse_env
eye_mouse_env\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
eye_mouse_env\Scripts\activate
python eye_mouse.py
```

**Calibration:** On startup, a red dot moves through the 4 corners of the screen. Hold your gaze on each dot for 1.5 seconds. Once all 4 points are captured, the cursor begins following your eyes.

- Press **R** to recalibrate at any time
- Press **Q** to quit

## Known Limitations / Work in Progress

This project is functional but still being refined:
- Cursor movement can be jittery, making precise clicks harder than intended
- The app occasionally crashes after calibration

Planned fixes include smoothing the cursor coordinate signal (moving average / exponential smoothing on iris position) and adding error handling around the calibration step. Tracking progress on these issues openly here rather than polishing only the parts that already work.

## Conclusion

This project demonstrates that assistive technology doesn't need to be expensive or rely on specialized hardware. With free libraries and a standard webcam, it's possible to build a functional hands-free input system for people who can't use conventional input devices — and there's still room to make it more robust.
