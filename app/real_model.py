"""
Real model integration: MediaPipe Hands -> 63 features -> MLP classifier.

This replaces mock_model.py. The contract is identical:
    predict(frame, target_letter=None) -> dict

The model file is named "svm_model.pkl" for historical reasons but actually
contains an MLPClassifier (512, 256) trained on 63 wrist-relative, scale-
normalized features. The interface (predict / predict_proba) is the same.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from letters import SPECIAL_CLASSES


# --------------------------------------------------------------------------
# Model loading (lazy singleton)
# --------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.pkl")

_model = None
_encoder = None


def _load_model():
    global _model, _encoder
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            _encoder = pickle.load(f)
    return _model, _encoder


# --------------------------------------------------------------------------
# Feature extraction — must match the training pipeline exactly
# --------------------------------------------------------------------------

def extract_features(landmarks) -> np.ndarray:
    """
    Convert MediaPipe Hands landmarks (21 points) into the 63-feature
    vector used by the classifier.

    Order: [x0..x20, y0..y20, z0..z20]
    Origin: wrist (landmark 0) — all coords are wrist-relative.
    Scale: divide by ||(x9, y9)|| — distance from wrist to middle-MCP.

    This matches the notebook exactly (data_collection + scale_normalize).
    """
    wrist = landmarks[0]
    xs = np.array([p.x - wrist.x for p in landmarks], dtype=np.float32)
    ys = np.array([p.y - wrist.y for p in landmarks], dtype=np.float32)
    zs = np.array([p.z - wrist.z for p in landmarks], dtype=np.float32)
    feats = np.concatenate([xs, ys, zs])

    # scale_normalize from notebook
    ref_x = feats[9]    # x9
    ref_y = feats[9 + 21]  # y9
    scale = np.sqrt(ref_x ** 2 + ref_y ** 2)
    if scale < 1e-6:
        scale = 1.0
    return feats / scale


# --------------------------------------------------------------------------
# Feedback hints (per letter — used when prediction is wrong)
# --------------------------------------------------------------------------

LETTER_HINTS = {
    "A": "Make a tight fist with your thumb on the side.",
    "B": "All four fingers straight up, thumb across the palm.",
    "C": "Curve your hand into a C shape, leave a clear gap.",
    "D": "Index up, three fingers touching the thumb in a circle.",
    "E": "Curl all fingers down, tuck the thumb under.",
    "F": "Thumb and index in a circle, three fingers up.",
    "G": "Index horizontal, thumb on top, other fingers curled.",
    "H": "Index and middle horizontal together.",
    "I": "Only the pinky goes up.",
    "K": "Index up, middle out, thumb touching the middle.",
    "L": "Right angle: thumb sideways, index straight up.",
    "M": "Three fingers over the thumb, knuckles rounded.",
    "N": "Two fingers over the thumb.",
    "O": "All fingers meet the thumb in a round O.",
    "P": "Like K but pointing down.",
    "Q": "Like G but pointing down.",
    "R": "Cross the middle finger over the index.",
    "S": "Fist with the thumb on top of the fingers.",
    "T": "Thumb pokes between index and middle in a fist.",
    "U": "Index and middle straight up, pressed together.",
    "V": "Index and middle up, spread apart — V shape.",
    "W": "Three fingers up and spread — W shape.",
    "X": "Bend the index into a hook, others in a fist.",
    "Y": "Thumb and pinky out, other fingers curled.",
}

GENERIC_HINTS = [
    "Hold your hand steady",
    "Show your palm to the camera",
    "Move a bit closer to the camera",
    "Make sure your whole hand is visible",
]


def _make_feedback(predicted: str, target: Optional[str], confidence: float) -> str:
    """Build a short user-facing feedback string."""
    if target is None:
        return f"Looks like {predicted}"

    if predicted == target and confidence >= 0.85:
        return random.choice(["Perfect!", "Nailed it!", "Great pose", "Spot on!"])

    if predicted == target:
        return "Almost there — hold steady"

    if predicted in SPECIAL_CLASSES:
        return f"That looks like a {predicted.lower()} sign — try letter {target}"

    hint = LETTER_HINTS.get(target, "Adjust your hand shape")
    return f"Looks more like {predicted}. For {target}: {hint}"


# --------------------------------------------------------------------------
# MediaPipe instance — one per VideoProcessor thread!
# --------------------------------------------------------------------------

def create_hands_detector():
    """
    Create a MediaPipe Hands instance.
    IMPORTANT: each thread needs its own instance — do NOT share across threads.
    """
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


# --------------------------------------------------------------------------
# Core prediction function — same contract as the mock
# --------------------------------------------------------------------------

def predict(
    frame: Optional[np.ndarray],
    target_letter: Optional[str] = None,
    hands_detector=None,
) -> dict:
    """
    Run prediction on a BGR frame.

    Args:
        frame: BGR image (np.ndarray) or None
        target_letter: the letter the user is supposed to show, for feedback
        hands_detector: optional pre-created MediaPipe Hands instance
                        (pass one to avoid recreating per call in webrtc)

    Returns:
        {
            "hand_detected": bool,
            "letter": str | None,        # predicted class, or None if no hand
            "confidence": float,         # 0..1
            "feedback": str,             # human-readable hint
        }
    """
    if frame is None:
        return {
            "hand_detected": False,
            "letter": None,
            "confidence": 0.0,
            "feedback": "No frame from camera",
        }

    model, encoder = _load_model()
    hands = hands_detector if hands_detector is not None else create_hands_detector()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return {
            "hand_detected": False,
            "letter": None,
            "confidence": 0.0,
            "feedback": random.choice([
                "Show your hand to the camera",
                "I don't see a hand — try again",
                "Move your hand into view",
            ]),
        }

    landmarks = result.multi_hand_landmarks[0].landmark
    features = extract_features(landmarks).reshape(1, -1)

    proba = model.predict_proba(features)[0]
    best_idx = int(np.argmax(proba))
    letter = encoder.inverse_transform([best_idx])[0]
    confidence = float(proba[best_idx])

    feedback = _make_feedback(letter, target_letter, confidence)

    return {
        "hand_detected": True,
        "letter": letter,
        "confidence": confidence,
        "feedback": feedback,
    }


def draw_landmarks_overlay(frame: np.ndarray, hands_detector) -> np.ndarray:
    """
    Draw MediaPipe hand landmarks on the frame for visual feedback.
    Returns a new frame with the overlay.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    if not result.multi_hand_landmarks:
        return frame

    out = frame.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        out,
        result.multi_hand_landmarks[0],
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(123, 212, 47), thickness=2, circle_radius=3),
        mp.solutions.drawing_utils.DrawingSpec(color=(255, 200, 59), thickness=2),
    )
    return out
