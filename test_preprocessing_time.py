import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import time

# ── Config ──────────────────────────────────────────────────────────────
DRIVE_TRAIN = Path('/Users/i2gunshaker/Library/CloudStorage/GoogleDrive-230183026@sdu.edu.kz/My Drive/asl_project/data/asl_alphabet_train/asl_alphabet_train')
LOCAL_TRAIN = Path('./asl_local/asl_alphabet_train')  # local ssd copy
MODEL_PATH = './hand_landmarker.task'

CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["space", "del", "nothing"]
FOLDER_MAP = {c: c for c in CLASSES}

CAP = 500  # max images per class
SEED = 42

OUTPUT_CSV = './landmarks_dataset1_benchmark.csv'

# ── Step 1: Copy dataset to local storage ───────────────────────────────
def copy_to_local():
    """Copy from Drive to local SSD for faster reads."""
    start_time = time.time()
    if LOCAL_TRAIN.exists():
        existing = len([d for d in LOCAL_TRAIN.iterdir() if d.is_dir()])
        if existing >= len(CLASSES):
            print(f"Local copy already exists ({existing} folders). Skipping copy.")
            return time.time() - start_time
        else:
            print(f"Incomplete local copy ({existing} folders). Re-copying...")
            shutil.rmtree(LOCAL_TRAIN.parent, ignore_errors=True)

    print("Copying dataset to local storage...")
    LOCAL_TRAIN.mkdir(parents=True, exist_ok=True)
    for cls in tqdm(CLASSES, desc="Copying classes"):
        src = DRIVE_TRAIN / FOLDER_MAP[cls]
        dst = LOCAL_TRAIN / FOLDER_MAP[cls]
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            
    copy_time = time.time() - start_time
    print(f"Local copy ready in {copy_time:.2f} seconds.\n")
    return copy_time


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

    total_start_time = time.time()

    # Set up MediaPipe
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    records = []
    stats = {}  # per-class stats

    print(f"Extracting landmarks | {len(CLASSES)} classes | CAP={CAP}\n")

    extract_start_time = time.time()
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for cls in CLASSES:
            folder = DRIVE_TRAIN / FOLDER_MAP[cls]
            if not folder.exists():
                print(f"  ⚠ {cls:>8s}: folder not found, skipping")
                stats[cls] = {'attempted': 0, 'extracted': 0}
                continue

            images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
            random.shuffle(images)
            images = images[:CAP]  # only process CAP images

            extracted = 0
            failed = 0

            # Normalize label
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

    extract_time = time.time() - extract_start_time

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    
    total_time = time.time() - total_start_time

    # Summary
    total_attempted = sum(s['attempted'] for s in stats.values())
    total_extracted = sum(s['extracted'] for s in stats.values())
    
    # Calculate images per second (extraction phase only)
    if extract_time > 0:
        ips_extraction = total_attempted / extract_time
    else:
        ips_extraction = 0
        
    print(f"\n{'='*50}")
    print(f"  Total extracting time: {extract_time:.2f} seconds")
    print(f"  Processing Speed:      {ips_extraction:.2f} images/second")
    print(f"  Total overall time:    {total_time:.2f} seconds")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
