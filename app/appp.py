"""
Znak — Streamlit app for learning the ASL alphabet.

Run:
    streamlit run app.py

Files:
    app.py             — entry point: state, sidebar, CSS, routing
    views.py           — page renderers
    real_model.py      — MediaPipe + MLP classifier (live model)
    letters.py         — alphabet metadata
    svm_model.pkl      — trained MLP (must be in this folder)
    label_encoder.pkl  — label encoder (must be in this folder)
"""

import json
import os
import streamlit as st

from views import (
    render_challenge,
    render_home,
    render_lesson,
    render_practice,
    render_word_mode,
)


# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="Znak — Learn sign language",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# Model file check — friendly error if pkls are missing
# ============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_FILES = ["best_model.pkl", "label_encoder.pkl"]
_missing = [f for f in _MODEL_FILES if not os.path.exists(os.path.join(_HERE, f))]
if _missing:
    st.error(
        f"⚠️ Model files not found: {', '.join(_missing)}\n\n"
        f"Place them next to `app.py` (folder: `{_HERE}`)."
    )
    st.stop()


# ============================================================================
# CSS — dark theme, Duolingo-inspired
# ============================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Fraunces:wght@600;800;900&display=swap');

:root {
    --bg-page: #0F1117;
    --bg-card: #1A1D24;
    --bg-card-hover: #22262F;
    --bg-elevated: #232730;
    --border: #2A2E38;
    --border-strong: #3A3F4B;

    --text-primary: #F1F3F5;
    --text-secondary: #9CA3AF;
    --text-muted: #6B7280;

    --primary: #58CC02;
    --primary-bright: #7BD42F;
    --primary-dark: #3F8F00;
    --primary-shadow: #2E6800;

    --secondary: #1CB0F6;
    --secondary-dark: #0E8DCB;

    --xp: #FFD43B;
    --xp-dark: #B8941F;
    --streak: #FF7B1C;
    --streak-dark: #B85A00;
    --heart: #FF5252;

    --locked: #2A2E38;
    --locked-dark: #1A1D24;
    --locked-text: #5A6072;

    --success-bg: rgba(88, 204, 2, 0.15);
    --success-border: rgba(88, 204, 2, 0.4);
    --error-bg: rgba(255, 82, 82, 0.15);
    --error-border: rgba(255, 82, 82, 0.4);
    --info-bg: rgba(28, 176, 246, 0.15);
    --info-border: rgba(28, 176, 246, 0.4);
}

/* Global */
html, body, [class*="css"], .stApp, .main, .block-container {
    font-family: 'Nunito', system-ui, sans-serif !important;
    color: var(--text-primary);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

.stApp {
    background:
        radial-gradient(ellipse at top left, rgba(88, 204, 2, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at top right, rgba(28, 176, 246, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at bottom, rgba(255, 212, 59, 0.04) 0%, transparent 60%),
        var(--bg-page);
    color: var(--text-primary);
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 980px !important;
}

/* Typography */
h1, h2, h3 {
    font-family: 'Fraunces', 'Nunito', serif !important;
    font-weight: 900 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

h1 { font-size: 2.5rem !important; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.75rem !important; }
h3 { font-size: 1.25rem !important; }

p, span, div {
    color: var(--text-primary);
}

.tagline {
    color: var(--text-secondary) !important;
    font-size: 1.05rem;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 1.5rem;
}

.tagline.subtle, .letter-desc.subtle {
    color: var(--text-muted) !important;
}

.section-title {
    margin-top: 2.5rem !important;
    margin-bottom: 1rem !important;
}

/* Hero */
.hero {
    text-align: center;
    padding: 1rem 0 0.5rem;
}

.hero-emoji {
    font-size: 5rem;
    line-height: 1;
    margin-bottom: 0.5rem;
    display: inline-block;
    animation: wiggle 3s ease-in-out infinite;
    transform-origin: 70% 70%;
    filter: drop-shadow(0 0 24px rgba(88, 204, 2, 0.4));
}

@keyframes wiggle {
    0%, 90%, 100% { transform: rotate(0deg); }
    93% { transform: rotate(-12deg); }
    96% { transform: rotate(12deg); }
}

.hero h1 {
    font-size: 3rem !important;
    margin: 0 !important;
    background: linear-gradient(90deg, var(--primary-bright) 0%, var(--secondary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero .tagline {
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}

/* Buttons — duolingo hard-shadow on dark */
.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.5rem !important;
    box-shadow: 0 4px 0 0 var(--primary-shadow) !important;
    transition: transform 0.08s, box-shadow 0.08s, background 0.15s !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    white-space: pre-line !important;
    line-height: 1.3 !important;
    min-height: 2.8rem;
}

.stButton > button:hover {
    background: var(--primary-bright) !important;
    color: white !important;
}

.stButton > button:active {
    transform: translateY(3px) !important;
    box-shadow: 0 1px 0 0 var(--primary-shadow) !important;
}

.stButton > button:focus {
    box-shadow: 0 4px 0 0 var(--primary-shadow) !important;
    outline: none !important;
}

/* Stat pills */
.stat-pills {
    display: flex;
    gap: 12px;
    margin: 1rem 0 1.5rem;
    flex-wrap: wrap;
}

.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 14px;
    padding: 0.5rem 1rem;
    font-weight: 800;
    font-size: 1.1rem;
    box-shadow: 0 2px 0 0 var(--border);
}

.stat-pill.streak { color: var(--streak); }
.stat-pill.xp { color: var(--xp); }
.stat-pill.heart { color: var(--heart); }

/* Cards */
.card {
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 18px;
    padding: 1.75rem;
    box-shadow: 0 2px 0 0 var(--border);
    margin: 1rem 0;
    color: var(--text-primary);
}

.progress-card { margin: 1rem 0 2rem; }
.progress-card h3 {
    margin-top: 0 !important;
    margin-bottom: 0.75rem !important;
}

.progress-bar {
    background: var(--bg-elevated);
    height: 16px;
    border-radius: 999px;
    overflow: hidden;
    position: relative;
    border: 1px solid var(--border);
}

.progress-bar-fill {
    background: linear-gradient(90deg, var(--primary) 0%, var(--primary-bright) 100%);
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease-out;
    box-shadow: 0 0 12px rgba(88, 204, 2, 0.5);
}

.progress-meta {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
    font-weight: 700;
    color: var(--text-secondary);
}

.progress-meta .pct {
    color: var(--primary-bright);
    font-weight: 900;
}

/* Levels */
.level-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px dashed var(--border);
}

.level-title {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--text-primary);
}

.level-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 600;
}

.level-counter {
    font-weight: 900;
    color: var(--primary-bright);
    font-size: 1.2rem;
}

.locked-msg {
    color: var(--text-muted);
    font-style: italic;
    font-weight: 600;
    margin: 0.5rem 0;
}

.letter-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0.75rem 0;
}

.letter-circle {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fraunces', serif;
    font-weight: 900;
    font-size: 1.5rem;
    flex-shrink: 0;
}

.letter-circle.unlocked {
    background: var(--primary);
    color: white;
    box-shadow: 0 4px 0 0 var(--primary-shadow);
}

.letter-circle.completed {
    background: var(--xp);
    color: #2A1F00;
    box-shadow: 0 4px 0 0 var(--xp-dark);
    font-size: 1.7rem;
}

.letter-circle.locked {
    background: var(--locked);
    color: var(--locked-text);
    box-shadow: 0 4px 0 0 var(--locked-dark);
}

/* Big letter card */
.letter-card {
    text-align: center;
    padding: 2rem 1.5rem !important;
}

.letter-label {
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 800;
}

.big-letter {
    font-family: 'Fraunces', serif;
    font-size: 8rem;
    font-weight: 900;
    color: var(--primary-bright);
    line-height: 1;
    margin: 0.5rem 0;
    text-shadow: 0 0 30px rgba(123, 212, 47, 0.4);
}

.big-emoji {
    font-size: 4rem;
    margin: 0.5rem 0;
}

.big-emoji-center {
    font-size: 5rem;
    text-align: center;
    margin: 0.5rem 0;
}

.letter-desc {
    font-size: 1rem;
    color: var(--text-secondary);
    line-height: 1.5;
    font-weight: 600;
    margin-top: 1rem;
}

/* Feedback messages */
.feedback {
    padding: 1rem 1.25rem;
    border-radius: 14px;
    font-weight: 700;
    margin: 1rem 0;
    border: 2px solid;
    font-size: 1rem;
}

.feedback.success {
    background: var(--success-bg);
    border-color: var(--success-border);
    color: var(--primary-bright);
}

.feedback.error {
    background: var(--error-bg);
    border-color: var(--error-border);
    color: var(--heart);
}

.feedback.info {
    background: var(--info-bg);
    border-color: var(--info-border);
    color: var(--secondary);
}

/* Lesson */
.lesson-progress {
    margin-top: 0.7rem;
    height: 12px;
}

.lesson-counter {
    text-align: center;
    color: var(--text-secondary);
    font-weight: 700;
    margin-top: 0.5rem;
}

.lesson-counter b {
    color: var(--primary-bright);
}

.lesson-complete {
    background: linear-gradient(135deg, rgba(255, 212, 59, 0.2) 0%, rgba(255, 123, 28, 0.15) 100%);
    border: 2px solid var(--xp);
    color: var(--xp);
    padding: 1.25rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.3rem;
    margin: 1.5rem 0;
    box-shadow: 0 0 24px rgba(255, 212, 59, 0.2);
    font-weight: 800;
}

/* Tips */
.tip-card {
    background: var(--info-bg);
    border: 2px solid var(--info-border);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    font-weight: 700;
    color: var(--secondary);
    height: 100%;
}

/* Practice */
.practice-stats {
    display: flex;
    gap: 12px;
    margin-top: 1rem;
}

.big-stat {
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    text-align: center;
    flex: 1;
    box-shadow: 0 2px 0 0 var(--border);
}

.big-stat .value {
    font-family: 'Fraunces', serif;
    font-size: 2.25rem;
    font-weight: 900;
    line-height: 1;
    color: var(--primary-bright);
}

.big-stat .label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.4rem;
    font-weight: 800;
}

/* Challenge */
.challenge-start, .challenge-end {
    text-align: center;
    padding: 2.5rem !important;
    max-width: 480px;
    margin: 1rem auto !important;
}

.challenge-final {
    display: flex;
    gap: 12px;
    margin-top: 1.5rem;
}

.challenge-hud {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin: 1rem 0 0.5rem;
}

.hud-cell {
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-weight: 900;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 0 0 var(--border);
}

.hud-cell.hud-hearts {
    font-size: 1.25rem;
    letter-spacing: 4px;
}

.timer-bar {
    height: 8px !important;
    margin-bottom: 1rem;
}

.timer-fill {
    background: linear-gradient(90deg, var(--heart) 0%, var(--streak) 50%, var(--xp) 100%) !important;
    box-shadow: 0 0 12px rgba(255, 123, 28, 0.4) !important;
}

/* Words */
.word-display {
    font-family: 'Fraunces', serif;
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem;
    line-height: 1.2;
}

.word-display .done {
    color: var(--primary-bright);
    text-shadow: 0 0 20px rgba(123, 212, 47, 0.4);
}

.word-display .current {
    color: var(--xp);
    border-bottom: 6px solid var(--xp);
    padding-bottom: 4px;
    animation: pulse-current 1.2s ease-in-out infinite;
    text-shadow: 0 0 20px rgba(255, 212, 59, 0.5);
}

.word-display .pending {
    color: var(--locked-text);
}

.word-display .space {
    display: inline-block;
    width: 0.5em;
}

@keyframes pulse-current {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Sidebar (dark) */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 2px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] h2 {
    font-family: 'Fraunces', serif !important;
}

[data-testid="stSidebar"] [data-baseweb="radio"] label {
    color: var(--text-primary) !important;
}

/* Radio in sidebar */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.4rem 0;
    cursor: pointer;
}

/* Live status indicator (under camera) */
.live-status {
    padding: 0.7rem 1rem;
    border-radius: 12px;
    font-weight: 700;
    margin: 0.5rem 0;
    border: 2px solid var(--border);
    background: var(--bg-card);
    text-align: center;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

.live-status.active {
    border-color: var(--primary);
    background: var(--success-bg);
}

.live-status.active .label {
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.live-status.active .val {
    color: var(--primary-bright);
    font-family: 'Fraunces', serif;
    font-size: 1.8rem;
    font-weight: 900;
    line-height: 1;
}

.live-status.active .conf {
    color: var(--xp);
    font-weight: 800;
}

.live-status.idle {
    color: var(--text-secondary);
}

.live-status.warning {
    background: var(--info-bg);
    border-color: var(--info-border);
    color: var(--secondary);
    text-align: left;
    font-size: 0.9rem;
}

/* webrtc video container styling */
[data-testid="stWebrtcStreamer"] video {
    border-radius: 16px;
    border: 2px solid var(--border);
    box-shadow: 0 4px 0 0 var(--border);
    width: 100% !important;
    background: var(--bg-card);
}

[data-testid="stWebrtcStreamer"] button {
    text-transform: uppercase !important;
    font-weight: 800 !important;
}

/* Columns padding */
[data-testid="column"] {
    padding: 0 0.5rem;
}

/* Mobile */
@media (max-width: 768px) {
    .big-letter { font-size: 5rem; }
    .word-display { font-size: 2.5rem; }
    .hero h1 { font-size: 2rem !important; }
    .hero-emoji { font-size: 3.5rem; }
}

/* Streamlit error/warning containers */
.stAlert {
    background: var(--bg-card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
# Session state init
# ============================================================================

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.json")


def save_progress():
    s = st.session_state
    try:
        data = {
            "xp": s.get("xp", 0),
            "streak": s.get("streak", 1),
            "completed_letters": list(s.get("completed_letters", set())),
            "letter_stats": s.get("letter_stats", {}),
            "practice_correct": s.get("practice_correct", 0),
            "practice_total": s.get("practice_total", 0),
        }
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_progress():
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
            data["completed_letters"] = set(data.get("completed_letters", []))
            return data
    except Exception:
        pass
    return {}


def init_state():
    defaults = {
        "page": "home",
        "xp": 0,
        "streak": 1,
        "hearts": 5,
        "completed_letters": set(),
        "letter_stats": {},
        "current_letter": None,
        "lesson_correct": 0,
        "last_prediction": None,
        "last_error": None,
        "practice_letter": None,
        "practice_correct": 0,
        "practice_total": 0,
        "challenge_active": False,
        "challenge_score": 0,
        "challenge_mistakes": 0,
        "challenge_letter": None,
        "challenge_start_ts": 0.0,
        "word_target": None,
        "word_index": 0,
        "word_done": False,
    }
    saved = load_progress()
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = saved.get(k, v)


init_state()


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## 🤟 Znak")
    st.markdown("*ASL learning app*")
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        options=["home", "lesson", "practice", "challenge", "word"],
        format_func=lambda p: {
            "home": "🏠  Home",
            "lesson": "📚  Lesson",
            "practice": "🎯  Practice",
            "challenge": "⚡  Challenge",
            "word": "📝  Words",
        }[p],
        index=["home", "lesson", "practice", "challenge", "word"].index(
            st.session_state.page
        ),
        label_visibility="collapsed",
    )
    if nav != st.session_state.page:
        st.session_state.page = nav
        st.rerun()

    st.markdown("---")
    st.markdown("**Dev tools**")

    if st.button("🔄  Reset progress", key="reset_progress"):
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.button("⚡  +50 XP", key="add_xp"):
        st.session_state.xp += 50
        st.rerun()

    if st.button("✅  Unlock all", key="complete_all"):
        from letters import ALPHABET as _A
        st.session_state.completed_letters = set(_A)
        st.rerun()


# ============================================================================
# Routing
# ============================================================================

ROUTES = {
    "home": render_home,
    "lesson": render_lesson,
    "practice": render_practice,
    "challenge": render_challenge,
    "word": render_word_mode,
}

ROUTES.get(st.session_state.page, render_home)()
save_progress()
