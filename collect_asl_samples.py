import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import urllib.request
import ssl

CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["SPACE", "DELETE", "NOTHING"]
SAMPLES_PER_CLASS = 100
OUTPUT_DIR = "asl_samples"
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

COUNTDOWN_SECONDS = 3.0
TIME_BETWEEN_SAMPLES = 0.1  # 10fps — gives natural variation between frames


def download_model():
    min_size = 1_000_000
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < min_size:
        os.remove(MODEL_PATH)
        print("Cached model was incomplete, re-downloading...")

    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8MB)...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())
        size = os.path.getsize(MODEL_PATH)
        if size < min_size:
            os.remove(MODEL_PATH)
            raise RuntimeError(f"Download failed ({size} bytes) — check your internet and retry.")
        print(f"Model ready ({size // 1024 // 1024}MB).")


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def already_collected(class_name):
    folder = os.path.join(OUTPUT_DIR, class_name)
    if not os.path.exists(folder):
        return False
    # exclude hidden files like .DS_Store on Mac
    images = [f for f in os.listdir(folder) if not f.startswith('.')]
    return len(images) >= SAMPLES_PER_CLASS


def collect_class(cap, landmarker, class_name):
    folder = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(folder, exist_ok=True)

    count = 0
    state = "WAITING"
    countdown_start = 0.0
    last_save = 0.0
    auto_paused = False  # tracks whether pause was triggered by hand leaving frame

    requires_hand = class_name != "NOTHING"

    print(f"\nReady for: {class_name}  |  Press SPACEBAR to start countdown.")
    if not requires_hand:
        print("  NOTHING class: keep hand out of frame.")
    print("  Controls: P = pause/resume | Q = quit")

    while count < SAMPLES_PER_CLASS:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed — check your webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        hand_detected = len(result.hand_landmarks) > 0

        clean_frame = frame.copy()

        if hand_detected and requires_hand:
            draw_landmarks(frame, result.hand_landmarks[0])

        # warn if hand appears during NOTHING recording
        if not requires_hand and hand_detected and state == "RECORDING":
            cv2.putText(frame, "WARNING: Move hand out of frame",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        border_color = (80, 80, 80)
        if state == "RECORDING":
            border_color = (0, 200, 0)
        elif state == "COUNTDOWN":
            border_color = (0, 165, 255)
        elif state == "PAUSED":
            border_color = (0, 0, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), border_color, 6)

        status = f"Class: {class_name}  |  Saved: {count}/{SAMPLES_PER_CLASS}"
        if requires_hand and not hand_detected:
            status += "  |  NO HAND DETECTED"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if state == "WAITING":
            cv2.putText(frame, "Press SPACE to start", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        elif state == "PAUSED":
            label = "PAUSED (hand left frame) — reposition and press SPACE" if auto_paused else "PAUSED — press P or SPACE to resume"
            cv2.putText(frame, label, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        elif state == "COUNTDOWN":
            time_left = COUNTDOWN_SECONDS - (time.time() - countdown_start)
            if time_left <= 0:
                state = "RECORDING"
                last_save = time.time() - TIME_BETWEEN_SAMPLES
                print(f"  Recording {class_name}...")
            else:
                remaining = max(1, min(int(time_left) + 1, int(COUNTDOWN_SECONDS)))
                cv2.putText(frame, str(remaining),
                            (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 165, 255), 5)

        cv2.imshow("ASL Sample Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quit early.")
            return False

        elif key in (ord(' '), ord('p')):
            if state in ("WAITING", "PAUSED"):
                if requires_hand and not hand_detected:
                    print("  No hand detected — move your hand into frame first.")
                elif auto_paused:
                    # hand left frame mid-recording — skip countdown, resume immediately
                    state = "RECORDING"
                    auto_paused = False
                    last_save = time.time() - TIME_BETWEEN_SAMPLES
                    print("  Resuming...")
                else:
                    state = "COUNTDOWN"
                    countdown_start = time.time()
                    auto_paused = False
            elif state == "RECORDING" and key == ord('p'):
                state = "PAUSED"
                auto_paused = False
                print("  Paused. Press P or SPACE to resume.")

        if state == "RECORDING":
            if requires_hand and not hand_detected:
                state = "PAUSED"
                auto_paused = True
                print("  Hand left frame — reposition and press SPACE to resume.")
            else:
                now = time.time()
                if now - last_save >= TIME_BETWEEN_SAMPLES:
                    filename = os.path.join(folder, f"{class_name}_{count:03d}.jpg")
                    cv2.imwrite(filename, clean_frame)
                    count += 1
                    last_save = now

    print(f"  Done with {class_name} ({count} images saved)")
    return True


def main():
    download_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pending = [c for c in CLASSES if not already_collected(c)]
    if not pending:
        print("All classes already collected. Delete subfolders in asl_samples/ to redo any class.")
        return

    skipped = [c for c in CLASSES if already_collected(c)]
    if skipped:
        print(f"Skipping already collected: {' '.join(skipped)}")

    print(f"\nClasses to collect: {' '.join(pending)}")
    print("Controls: SPACE = start | P = pause/resume | Q = quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera. Make sure no other app is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(30):
        cap.read()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for class_name in pending:
            ok = collect_class(cap, landmarker, class_name)
            if not ok:
                break

    cap.release()
    cv2.destroyAllWindows()

    collected = [c for c in CLASSES if already_collected(c)]
    print(f"\nDone! Collected {len(collected)}/{len(CLASSES)} classes.")
    print(f"Images saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("Zip the asl_samples/ folder and upload it to the team shared folder.")


if __name__ == "__main__":
    main()
