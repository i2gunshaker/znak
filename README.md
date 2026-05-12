# 🤟 Znak — ASL Fingerspelling Trainer

Real-time American Sign Language alphabet trainer. MediaPipe extracts hand landmarks from your webcam, an MLP classifier predicts the sign, and a Duolingo-style UI tracks your progress.

**[Live demo →](https://asl-learning-app-slise-of-life.streamlit.app)**

---

## Dataset

Two sources combined into one dataset:

**Kaggle — [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)** (grassknoted)
87,000 images across 27 classes. MediaPipe extracted landmarks from a 1,000-image sample per class → 19,592 rows after dropout (27.4% of images had no detectable hand).

**Self-collected** — 4 contributors recorded their own samples using `collect_asl_samples.py`. The script captures 100 frames per class through a live webcam view, only saving frames where MediaPipe detects a hand.

| source | rows |
|---|---|
| kaggle_alphabet | 19,592 |
| self_collected_akezhun | 5,200 |
| self_collected_pavel / pasha | 3,900 |
| self_collected_akira | 2,600 |
| self_collected_i2 | 2,600 |
| **total** | **33,892** |

26 classes: A–Y (no J or Z — motion-based signs), SPACE, DELETE.

---

## Pipeline

```
image / webcam frame
    ↓
MediaPipe Hands → 21 landmarks (x, y, z)
    ↓
wrist-relative: subtract landmark 0
    ↓
flatten → [x0..x20, y0..y20, z0..z20] = 63 features
    ↓
scale normalize: divide by √(x9² + y9²)
    ↓
MLPClassifier → 26 class probabilities → letter
```

Scale normalization makes the feature vector invariant to hand size and camera distance.

---

## Models

Six classifiers trained and tuned. CV on original split, winner refit on augmented data (rotation ±15°, Gaussian noise σ=0.01, random x-flip — 27k → 81k samples).

| Model | Test accuracy |
|---|---|
| Logistic Regression | 94.45% |
| Random Forest | 97.46% |
| SVM | 97.52% |
| KNN | 98.33% |
| LightGBM | 98.33% |
| **MLP (512→256)** | **98.86%** |

Best params: `hidden_layer_sizes=(512, 256)`, `alpha=0.0001`, `learning_rate_init=0.001`.

---

## Repo structure

```
znak/
├── data/
│   └── landmarks_merged.csv       # 33,892 rows, 63 features + letter + source
├── notebooks/
│   ├── 01_extraction.ipynb        # Kaggle download + landmark extraction
│   ├── 02_eda.ipynb               # exploratory analysis
│   └── 03_training.ipynb          # training, tuning, evaluation
├── data_collection/
│   ├── collect_asl_samples.py     # webcam sample collection script
│   ├── extract_landmarks_samples.py
│   ├── extract_landmarks_dataset1.py
│   ├── INSTRUCTIONS_DataCollection.md
│   └── INSTRUCTIONS_LandmarkExtraction.md
├── app/
│   ├── appp.py                    # entry point: routing, CSS, session state
│   ├── views.py                   # pages: home, lesson, practice, challenge, words
│   ├── real_model.py              # MediaPipe + MLP inference
│   ├── letters.py                 # alphabet metadata, levels, word lists
│   ├── best_model.pkl             # trained MLP (98.86% test accuracy)
│   ├── label_encoder.pkl
│   └── packages.txt
├── requirements.txt
└── .gitignore
```

---

## Run locally

```bash
conda create -n asl python=3.11 -y
conda activate asl
pip install -r requirements.txt

cd app
streamlit run appp.py                                            # Windows / Linux
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES streamlit run appp.py   # macOS
```
