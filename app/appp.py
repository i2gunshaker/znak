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
from pathlib import Path
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

CSS_PATH = Path(__file__).parent / "styles.css"
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


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
