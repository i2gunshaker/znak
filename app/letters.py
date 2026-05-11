"""ASL alphabet metadata and curriculum structure.

The model recognizes 26 classes: 24 letters + DELETE + SPACE.
J and Z are not in the model (they are dynamic gestures in ASL).
DELETE and SPACE are excluded from the learning curriculum but the
model still recognizes them — we just don't teach them.
"""

# Canonical letter order — matches label_encoder.classes_ (minus DELETE/SPACE)
ALPHABET = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 letters, no J no Z

# Non-letter classes the model knows but we don't teach
SPECIAL_CLASSES = {"DELETE", "SPACE"}

LETTERS = {
    "A": {
        "description": "Make a fist with your thumb resting on the side.",
        "tips": ["Thumb straight along the side", "Knuckles facing forward", "Fist not too tight"],
        "emoji": "✊",
    },
    "B": {
        "description": "Four fingers straight up, pressed together. Thumb tucked across the palm.",
        "tips": ["Fingers tight together", "Thumb across the palm", "Palm facing forward"],
        "emoji": "✋",
    },
    "C": {
        "description": "Curve your hand into the shape of the letter C.",
        "tips": ["Fingers form a curve", "Thumb opposite the fingers", "Opening shaped like C"],
        "emoji": "🤏",
    },
    "D": {
        "description": "Index finger up. Other three fingers touch the thumb forming a circle.",
        "tips": ["Index pointing straight up", "Three fingers touch thumb", "Tight circle"],
        "emoji": "☝️",
    },
    "E": {
        "description": "All fingers curled down to the palm, thumb tucked under them.",
        "tips": ["Fingers curled, not flat", "Thumb beneath fingers", "Hand looks compact"],
        "emoji": "🤚",
    },
    "F": {
        "description": "Thumb and index touch in a circle. Other three fingers stick up.",
        "tips": ["Circle from thumb and index", "Three fingers straight up", "Palm forward"],
        "emoji": "👌",
    },
    "G": {
        "description": "Index finger extended horizontally, thumb parallel to it.",
        "tips": ["Index horizontal", "Thumb above index", "Other fingers curled in"],
        "emoji": "👉",
    },
    "H": {
        "description": "Index and middle fingers extended horizontally together.",
        "tips": ["Two fingers horizontal", "Fingers tight together", "Thumb tucked under"],
        "emoji": "🫳",
    },
    "I": {
        "description": "Pinky finger up. Other fingers curled into a fist.",
        "tips": ["Pinky straight up", "Other fingers curled", "Thumb on the side"],
        "emoji": "🤙",
    },
    "K": {
        "description": "Index up, middle finger angled out, thumb between them.",
        "tips": ["Index straight up", "Middle at 45 degrees", "Thumb touches middle"],
        "emoji": "✌️",
    },
    "L": {
        "description": "Thumb out to the side, index pointing up — forms an L shape.",
        "tips": ["Right angle between thumb and index", "Thumb horizontal", "Index vertical"],
        "emoji": "🫵",
    },
    "M": {
        "description": "Thumb tucked under three fingers (index, middle, ring).",
        "tips": ["Three fingers over thumb", "Knuckles rounded", "Pinky tucked in"],
        "emoji": "✊",
    },
    "N": {
        "description": "Thumb tucked under two fingers (index and middle).",
        "tips": ["Two fingers over thumb", "Ring and pinky tucked", "Knuckles visible"],
        "emoji": "✊",
    },
    "O": {
        "description": "All fingers curl to meet the thumb, forming a round O.",
        "tips": ["Tight circle, no gap", "Fingers move as one", "Side of hand to camera"],
        "emoji": "👌",
    },
    "P": {
        "description": "Like K, but pointing down: index down, middle angled out, thumb between.",
        "tips": ["Hand points downward", "Index straight down", "Thumb touches middle"],
        "emoji": "👇",
    },
    "Q": {
        "description": "Like G, but pointing down: index and thumb point down together.",
        "tips": ["Hand points downward", "Index pointing down", "Thumb parallel"],
        "emoji": "👇",
    },
    "R": {
        "description": "Index and middle fingers crossed.",
        "tips": ["Middle crosses over index", "Fingers straight up", "Other fingers curled"],
        "emoji": "🤞",
    },
    "S": {
        "description": "Fist with thumb wrapped over the front of the fingers.",
        "tips": ["Thumb on top of fingers", "Tight fist", "Knuckles forward"],
        "emoji": "✊",
    },
    "T": {
        "description": "Thumb pokes out between index and middle finger of a fist.",
        "tips": ["Thumb peeks between fingers", "Tight fist", "Rounded knuckles"],
        "emoji": "✊",
    },
    "U": {
        "description": "Index and middle fingers up, pressed together.",
        "tips": ["Two fingers tight together", "Straight up", "Thumb tucked in"],
        "emoji": "✌️",
    },
    "V": {
        "description": "Index and middle fingers up, spread apart — peace sign.",
        "tips": ["Fingers spread like V", "Straight up", "Other fingers curled"],
        "emoji": "✌️",
    },
    "W": {
        "description": "Index, middle, and ring fingers up and spread.",
        "tips": ["Three fingers like W", "Fingers spread apart", "Thumb touches pinky"],
        "emoji": "🖖",
    },
    "X": {
        "description": "Index finger bent like a hook, other fingers in a fist.",
        "tips": ["Index curled into hook", "Other fingers in fist", "Thumb on side"],
        "emoji": "☝️",
    },
    "Y": {
        "description": "Thumb and pinky extended, others curled — the hang-loose sign.",
        "tips": ["Thumb and pinky straight", "Other fingers curled", "Hang-loose vibe"],
        "emoji": "🤙",
    },
}

LEVELS = [
    {"name": "Level 1", "subtitle": "Simple shapes", "letters": ["A", "B", "C", "D", "L", "O", "Y"]},
    {"name": "Level 2", "subtitle": "Vowels and basics", "letters": ["E", "F", "I", "U", "V", "W"]},
    {"name": "Level 3", "subtitle": "A bit trickier", "letters": ["G", "H", "K", "P", "Q"]},
    {"name": "Level 4", "subtitle": "Look-alikes", "letters": ["M", "N", "R", "S", "T", "X"]},
]

WORDS_BY_DIFFICULTY = {
    "easy": ["CAT", "DOG", "BAD", "BAY", "OLD", "BOY", "BOW", "ABC"],
    "medium": ["HELLO", "WORLD", "CODE", "LOVE", "SMILE", "STAR", "FIRE", "BLUE"],
    "hard": ["ASTANA", "PYTHON", "STREAMLIT", "SCIENCE", "FUTURE", "DREAM"],
}
