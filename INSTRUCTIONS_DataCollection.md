# ASL Sample Collection — Instructions for Team Members

Each team member needs to record **100 images per class** across **27 classes**:
- **24 letters:** A–Y (no J or Z — these are motion-based signs)
- **3 special classes:** SPACE, DELETE, NOTHING

That's **27 classes × 100 images = 2,700 images per person.**

---

## Step 1: Install Python & Required Libraries

Make sure you have Python 3.8 or newer. Open a terminal (Mac/Linux) or Command Prompt (Windows) and run:

```
pip install opencv-python mediapipe numpy
```

If you get a permissions error on Mac/Linux, try:
```
pip install opencv-python mediapipe numpy --user
```

---

## Step 2: Download the Script

Get the file `collect_asl_samples.py` from the team shared folder (same place as these instructions).
Put it anywhere you like — your Desktop is fine.

---

## Step 3: Run the Script

Open a terminal, navigate to where you saved the script, and run:

```
python collect_asl_samples.py
```

Your webcam window will open. You'll see a live view of yourself.

---

## Step 4: Recording Your Samples

The script works **one class at a time**. Here's the flow:

1. **Look at the terminal** — it will tell you which class is up next (e.g., `Ready for: A`)
2. **Position your hand** in the camera frame — make the ASL sign for that class
3. **Press SPACEBAR** to start a 3-second countdown. The border turns orange during countdown.
4. **Hold the sign** while 100 photos are taken automatically (takes about 10 seconds)
5. The border turns back to grey — that class is done
6. **Move to the next class** — press SPACEBAR again when ready

### Controls
| Key | Action |
|-----|--------|
| SPACE | Start recording / Resume after pause |
| P | Pause / Resume recording |
| Q | Quit (progress is saved — rerun to continue) |

### Auto-pause
If your hand leaves the frame mid-recording, the script **automatically pauses** and waits for you to reposition. Press SPACE to resume — no countdown needed, it picks up right where you left off.

---

## Step 5: Special Classes (SPACE, DELETE, NOTHING)

These three classes come after the letters. Here's how to record them:

| Class | What to Sign | Notes |
|-------|-------------|-------|
| **SPACE** | Open palm facing camera, fingers spread | Used to insert a space between words in the demo |
| **DELETE** | Closed fist or pinching gesture | Used to delete the last character in the demo |
| **NOTHING** | Keep your hand **out of frame** entirely | Used for noise rejection — the model learns to ignore empty frames |

> ⚠️ For the **NOTHING** class, make sure no hand is visible. If a hand is detected while recording NOTHING, you'll see a red warning on screen.

---

## Step 6: Tips for Good Data

These tips directly affect model accuracy, so please follow them:

- **Vary your hand slightly** between shots — small rotations, different distances from camera. Don't hold perfectly still; slight natural variation is better for training.
- **Try 2–3 different lighting conditions** across classes (e.g., some near a window, some in normal room light). You don't need to redo all classes, just vary it naturally as you go.
- **Use your dominant hand** throughout. If you're left-handed, record everything left-handed — that's fine.
- **Keep your hand fully in frame** — the detector needs to see your entire hand including wrist.
- **Plain background preferred** but not required. A cluttered background is actually useful for robustness.
- **Do NOT attempt J or Z** — these are motion-based signs and cannot be captured in a single image. They are excluded from the project.

---

## Step 7: Share Your Data

After recording, a folder called `asl_samples/` will appear next to the script.
Inside it will be subfolders like `asl_samples/A/`, `asl_samples/B/`, ..., `asl_samples/SPACE/`, `asl_samples/DELETE/`, `asl_samples/NOTHING/`.

**Zip the entire `asl_samples/` folder** and upload it to the team shared Google Drive folder.
Name it with your name: e.g., `samples_omar.zip`

---

## Troubleshooting

**"No camera found" error:**
Make sure no other app (Zoom, Teams) is using your webcam. Close them and retry.

**Hand not being detected:**
Make sure your hand is well-lit and fully visible. The script requires MediaPipe to detect a hand before saving a photo — if no hand is detected, it will skip the frame and warn you.

**Script crashes on import:**
Run `pip install opencv-python mediapipe numpy` again. If on Mac with Apple Silicon, try `pip install opencv-python-headless` instead of `opencv-python`.

**I accidentally recorded a wrong sign:**
Delete the subfolder for that class (`asl_samples/B/` for example) and rerun — the script will redo missing classes automatically.

**I need to stop and continue later:**
Just press Q. The script saves progress per class — when you rerun, it skips any class that already has 100 images.

---

## Classes to Record

```
Letters:  A  B  C  D  E  F  G  H  I  K  L  M  N  O  P  Q  R  S  T  U  V  W  X  Y
Special:  SPACE  DELETE  NOTHING
```
(J and Z are excluded — motion signs)

**Total: 27 classes × 100 images = 2,700 images**

---

*If anything is unclear, message the group chat before guessing — consistent data collection matters.*
