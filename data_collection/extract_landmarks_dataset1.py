"""
Extract hand landmarks from the Kaggle ASL Alphabet dataset.
Outputs a CSV with 63 wrist-normalized features per sample.

Usage (in Colab):
  1. Mount Google Drive
  2. Run this script — it copies data locally first, then extracts

Classes: 24 letters (A-Y, no J/Z) + SPACE + DELETE + NOTHING = 27 total
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
DRIVE_ROOT = Path('/content/drive/MyDrive/asl_project/data/')
DRIVE_TRAIN = DRIVE_ROOT / 'asl_alphabet_train/asl_alphabet_train'
LOCAL_TRAIN = Path('/content/asl_local/asl_alphabet_train')  # fast SSD copy
MODEL_PATH = '/content/hand_landmarker.task'

CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["space", "del", "nothing"]
# Kaggle folder names: letters are uppercase, specials are lowercase
FOLDER_MAP = {c: c for c in CLASSES}

CAP = 500  # max images per class
SEED = 42

OUTPUT_CSV = DRIVE_ROOT / 'landmarks_dataset1.csv'

# ── Step 1: Copy dataset to local storage ───────────────────────────────
def copy_to_local():
    """Copy from Drive to Colab's local SSD for ~5x faster reads."""
    if LOCAL_TRAIN.exists():
        # count folders to check if copy is complete
        existing = len([d for d in LOCAL_TRAIN.iterdir() if d.is_dir()])
        if existing >= len(CLASSES):
            print(f"Local copy already exists ({existing} folders). Skipping copy.")
            return
        else:
            print(f"Incomplete local copy ({existing} folders). Re-copying...")
            shutil.rmtree(LOCAL_TRAIN.parent)

    print("Copying dataset to local storage (this takes ~2 min)...")
    # Only copy the folders we need, not the entire dataset
    LOCAL_TRAIN.mkdir(parents=True, exist_ok=True)
    for cls in tqdm(CLASSES, desc="Copying classes"):
        src = DRIVE_TRAIN / FOLDER_MAP[cls]
        dst = LOCAL_TRAIN / FOLDER_MAP[cls]
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
    print("Local copy ready.\n")


# ── Step 2: Extract landmarks ──────────────────────────────────────────
def extract_landmarks(img_path, landmarker):
    """Run MediaPipe on one image, return wrist-normalized landmark dict or None."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    row = {}
    for i, pt in enumerate(lm):
        row[f'x{i}'] = pt.x - wx
        row[f'y{i}'] = pt.y - wy
        row[f'z{i}'] = pt.z - wz
    return row


def main():
    random.seed(SEED)

    # Copy to local SSD
    copy_to_local()

    # Set up MediaPipe
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    records = []
    stats = {}  # per-class stats

    print(f"Extracting landmarks | {len(CLASSES)} classes | CAP={CAP}\n")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for cls in CLASSES:
            folder = LOCAL_TRAIN / FOLDER_MAP[cls]
            if not folder.exists():
                print(f"  ⚠ {cls:>8s}: folder not found, skipping")
                stats[cls] = {'attempted': 0, 'extracted': 0}
                continue

            images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
            random.shuffle(images)
            images = images[:CAP]  # only process CAP images

            extracted = 0
            failed = 0

            # Normalize label: map Kaggle folder names to our standard labels
            if cls == "space":
                label = "SPACE"
            elif cls == "del":
                label = "DELETE"
            elif cls == "nothing":
                label = "NOTHING"
            else:
                label = cls  # already uppercase letter

            for img_path in tqdm(images, desc=f"  {label:>8s}", leave=True):
                row = extract_landmarks(img_path, landmarker)
                if row is None:
                    failed += 1
                    continue
                row['letter'] = label
                row['source'] = 'kaggle_alphabet'
                records.append(row)
                extracted += 1

            stats[cls] = {'attempted': len(images), 'extracted': extracted, 'failed': failed}

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    # Summary
    total_attempted = sum(s['attempted'] for s in stats.values())
    total_extracted = sum(s['extracted'] for s in stats.values())
    total_failed = sum(s.get('failed', 0) for s in stats.values())

    print(f"\n{'='*50}")
    print(f"  Total extracted:  {total_extracted} / {total_attempted}")
    print(f"  Dropout:          {total_failed} ({total_failed/max(total_attempted,1)*100:.1f}%)")
    print(f"  Saved to:         {OUTPUT_CSV}")
    print(f"{'='*50}")

    # Per-class breakdown
    print(f"\n{'Class':>10s} {'Attempted':>10s} {'Extracted':>10s} {'Dropout%':>10s}")
    print("-" * 45)
    for cls in CLASSES:
        s = stats[cls]
        label = "SPACE" if cls == "space" else ("DELETE" if cls == "del" else ("NOTHING" if cls == "nothing" else cls))
        att = s['attempted']
        ext = s['extracted']
        fail = s.get('failed', 0)
        pct = f"{fail/att*100:.1f}%" if att > 0 else "N/A"
        print(f"{label:>10s} {att:>10d} {ext:>10d} {pct:>10s}")

    print(f"\nClass distribution:\n{df['letter'].value_counts().sort_index()}")


if __name__ == '__main__':
    main()
