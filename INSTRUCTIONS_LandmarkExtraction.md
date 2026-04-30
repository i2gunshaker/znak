# Landmark Extraction - Instructions

After you finish recording with `collect_asl_samples.py`, run this script to extract the hand landmarks from your images. It outputs a small CSV file that you upload to the team Drive. That's all we need from you - not the images.

---

## What this does

It runs MediaPipe on each photo you took and saves the hand joint positions as numbers. This is the actual input our model trains on. The images themselves are too large to share, so we share the CSV instead.

---

## Step 1: Put the files in the same folder

Make sure these three things are in the same folder:

```
collect_asl_samples.py       <- you already have this
extract_landmarks_samples.py <- new, download from team Drive
asl_samples/                 <- created when you ran collect_asl_samples.py
```

The `hand_landmarker.task` model file will be downloaded automatically if it's not there (same as the collection script).

---

## Step 2: Run it

```
python extract_landmarks_samples.py
```

It will ask for your name - type it in lowercase with no spaces (e.g. `omar`, `jane`, `alex`). This is just used to label your data in the CSV.

Then it processes all your images class by class. You'll see a progress bar for each one. It takes about 2-3 minutes total.

---

## Step 3: Upload the output

When it finishes you'll see something like:

```
Extracted: 2340 rows from 2700 images (360 dropped - hand not detected)
Saved: landmarks_omar.csv
```

Upload `landmarks_[yourname].csv` to the team Google Drive folder. That file is all we need.

---

## Notes

**Some images will be dropped** - that's normal. MediaPipe sometimes can't find a hand if the lighting is bad or the hand is partially out of frame. As long as you're getting most of your images through (say 70%+), you're fine.

**The NOTHING class will drop almost everything** - that's expected. NOTHING is photos with no hand in frame, so MediaPipe finds nothing to detect. That's fine, we handle it on our end.

**If the script crashes on import**, run this first:
```
pip install opencv-python mediapipe pandas tqdm
```

**If you get a low extraction rate on a specific letter** (like under 50%), it probably means your hand wasn't fully in frame for those shots. You can re-record just that class by deleting its subfolder (`asl_samples/B/` for example) and running `collect_asl_samples.py` again.

---

## Classes

```
Letters:  A B C D E F G H I K L M N O P Q R S T U V W X Y
Special:  SPACE  DELETE  NOTHING
```

(J and Z are excluded - motion signs, can't be captured in a single image)
