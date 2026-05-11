import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
import ssl
import os

SAMPLES_DIR = 'asl_samples'
MODEL_PATH = 'hand_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

CLASSES = list('ABCDEFGHIKLMNOPQRSTUVWXY') + ['SPACE', 'DELETE', 'NOTHING']


def download_model():
    min_size = 1_000_000
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= min_size:
        return
    print('Downloading hand landmark model (~8MB)...')
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(MODEL_URL, context=ctx) as r, open(MODEL_PATH, 'wb') as f:
        f.write(r.read())
    print('Done.\n')


def get_landmarks(img_path, detector):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None
    lm = res.hand_landmarks[0]
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z
    row = {}
    for i, pt in enumerate(lm):
        row[f'x{i}'] = pt.x - wx
        row[f'y{i}'] = pt.y - wy
        row[f'z{i}'] = pt.z - wz
    return row


def main():
    name = input('Enter your first name (no spaces, e.g. omar): ').strip().lower()
    if not name:
        print('Name cannot be empty.')
        return

    samples_dir = Path(SAMPLES_DIR)
    if not samples_dir.exists():
        print(f"Could not find '{SAMPLES_DIR}/' folder.")
        print("Make sure you ran collect_asl_samples.py first and this script is in the same folder.")
        return

    download_model()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    records = []
    attempted = 0
    dropped = 0

    print(f'\nExtracting landmarks from asl_samples/ ...\n')
    with vision.HandLandmarker.create_from_options(options) as detector:
        for cls in CLASSES:
            folder = samples_dir / cls
            if not folder.exists():
                continue
            imgs = [f for f in folder.glob('*.jpg')] + [f for f in folder.glob('*.png')]
            if not imgs:
                continue
            attempted += len(imgs)
            for path in tqdm(imgs, desc=f'{cls:>8}'):
                row = get_landmarks(path, detector)
                if row is None:
                    dropped += 1
                    continue
                row['letter'] = cls
                row['source'] = f'self_collected_{name}'
                records.append(row)

    if not records:
        print('No landmarks extracted. Make sure asl_samples/ has images inside.')
        return

    df = pd.DataFrame(records)
    out = f'landmarks_{name}.csv'
    df.to_csv(out, index=False)

    print(f'\nDone.')
    print(f'Extracted: {len(df)} rows from {attempted} images ({dropped} dropped - hand not detected)')
    print(f'Saved: {out}')
    print(f'\nUpload {out} to the team Google Drive folder.')
    print(f'\nBreakdown by class:')
    print(df['letter'].value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()
