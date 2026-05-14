"""
All page renderers. English UI, dark theme, real model via streamlit-webrtc.

Uses the `video_frame_callback` API (modern, works reliably with
streamlit-webrtc 0.47+). State is shared between the worker thread (running
the callback) and the main Streamlit thread via a module-level dict
protected by a lock.
"""

import collections
import logging
import random
import sys
import threading
import time
import traceback

import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from letters import ALPHABET, LETTERS, LEVELS, SPECIAL_CLASSES, WORDS_BY_DIFFICULTY
from real_model import create_hands_detector, predict


logger = logging.getLogger("znak")
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


REQUIRED_CORRECT = 3
CONF_THRESHOLD = 0.80
CHALLENGE_DURATION = 60.0


# ============================================================================
# Shared state — read/written by both UI thread and webrtc worker thread
# ============================================================================

_state_lock = threading.Lock()
_state = {
    "target_letter": None,
    "latest_prediction": None,
    "last_error": None,
    "frame_count": 0,
    "predict_count": 0,
}
_hands_detector = None
_hands_lock = threading.Lock()
_prediction_buffer = collections.deque(maxlen=7)


def _get_hands_detector():
    global _hands_detector
    with _hands_lock:
        if _hands_detector is None:
            _hands_detector = create_hands_detector()
            logger.info("MediaPipe Hands detector initialized")
    return _hands_detector


def set_target_letter(letter):
    with _state_lock:
        if _state["target_letter"] != letter:
            _prediction_buffer.clear()
        _state["target_letter"] = letter


def get_state_snapshot():
    with _state_lock:
        return {
            "prediction": dict(_state["latest_prediction"]) if _state["latest_prediction"] else None,
            "error": _state["last_error"],
            "frames": _state["frame_count"],
            "predicts": _state["predict_count"],
        }


def reset_state_error():
    with _state_lock:
        _state["last_error"] = None


# ============================================================================
# Frame callback — runs in webrtc's worker thread
# ============================================================================

def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        with _state_lock:
            target = _state["target_letter"]
            _state["frame_count"] += 1

        try:
            hands = _get_hands_detector()
        except Exception as exc:
            err = f"MediaPipe init failed: {exc}\n{traceback.format_exc()}"
            logger.error(err)
            with _state_lock:
                _state["last_error"] = err
            cv2.putText(img, "Model init failed - see terminal", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (80, 80, 220), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        result = predict(img, target_letter=target, hands_detector=hands)

        if result["hand_detected"]:
            _prediction_buffer.append(result["letter"])
            smoothed = collections.Counter(_prediction_buffer).most_common(1)[0][0]
            result = {**result, "letter": smoothed}
        else:
            _prediction_buffer.clear()

        with _state_lock:
            _state["latest_prediction"] = result
            _state["predict_count"] += 1
            _state["last_error"] = None

        if result["hand_detected"]:
            label = f"{result['letter']}  {result['confidence']:.0%}"
            if target is None or result["letter"] == target:
                color = (47, 212, 123)
            elif result["letter"] in SPECIAL_CLASSES:
                color = (200, 200, 200)
            else:
                color = (80, 80, 220)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            cv2.rectangle(img, (12, 12), (12 + tw + 24, 12 + th + 24), (20, 22, 30), -1)
            cv2.putText(img, label, (24, 12 + th + 12),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        else:
            cv2.putText(img, "Show your hand", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    except Exception as exc:
        err = f"callback error: {exc}\n{traceback.format_exc()}"
        logger.error(err)
        with _state_lock:
            _state["last_error"] = err
        try:
            img = frame.to_ndarray(format="bgr24")
            cv2.putText(img, f"ERROR: {str(exc)[:60]}", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (80, 80, 220), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame


# ============================================================================
# WebRTC streamer — TURN server routes around university/firewall P2P blocks
# ============================================================================

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}


def render_camera(target_letter, key):
    set_target_letter(target_letter)
    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        desired_playing_state=True,
    )
    return ctx


def grab_prediction(ctx):
    snap = get_state_snapshot()

    if snap["error"]:
        first_line = snap["error"].splitlines()[0]
        return None, f"Camera error: {first_line}"

    is_playing = bool(ctx and getattr(ctx.state, "playing", False))

    if not is_playing and snap["frames"] == 0:
        return None, "Camera is starting up…"

    if snap["prediction"] is None:
        if snap["frames"] == 0:
            return None, "Camera started but no frames received yet…"
        if snap["predicts"] == 0:
            return None, (
                f"Received {snap['frames']} frames but processed 0. "
                "Check the terminal for errors."
            )
        return None, "Waiting for first prediction…"

    return snap["prediction"], None


def render_live_status(ctx):
    pred, err = grab_prediction(ctx)

    if err:
        st.markdown(f"<div class='live-status warning'>⏳ {err}</div>",
                    unsafe_allow_html=True)
        return

    if pred and pred.get("hand_detected"):
        letter = pred["letter"]
        conf = pred["confidence"]
        st.markdown(
            f"""
            <div class='live-status active'>
                <span class='label'>Currently seeing:</span>
                <span class='val'>{letter}</span>
                <span class='conf'>{conf:.0%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif pred:
        st.markdown(
            "<div class='live-status idle'>👀 No hand in frame</div>",
            unsafe_allow_html=True,
        )


# ============================================================================
# Shared UI helpers
# ============================================================================

def render_stats():
    s = st.session_state
    st.markdown(
        f"""
        <div class="stat-pills">
            <span class="stat-pill streak">🔥 <span>{s.streak}</span></span>
            <span class="stat-pill xp">💎 <span>{s.xp}</span></span>
            <span class="stat-pill heart">❤️ <span>{s.hearts}</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def goto(page):
    st.session_state.page = page
    st.rerun()


def render_feedback(pred, target=None, error=None):
    if error:
        st.markdown(
            f"<div class='feedback info'>⏳ {error}</div>",
            unsafe_allow_html=True,
        )
        return

    if not pred:
        return

    if not pred["hand_detected"]:
        st.markdown(
            f"<div class='feedback info'>📷 {pred['feedback']}</div>",
            unsafe_allow_html=True,
        )
        return

    is_correct = (
        (target is None or pred["letter"] == target)
        and pred["confidence"] >= 0.85
        and pred["letter"] not in SPECIAL_CLASSES
    )

    if is_correct:
        st.markdown(
            f"<div class='feedback success'>✅ {pred['feedback']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='feedback error'>❌ {pred['feedback']}</div>",
            unsafe_allow_html=True,
        )


def _is_correct(pred, target):
    if not pred or not pred["hand_detected"]:
        return False
    if pred["letter"] in SPECIAL_CLASSES:
        return False
    return pred["letter"] == target and pred["confidence"] >= CONF_THRESHOLD


# ============================================================================
# HOME
# ============================================================================

def render_home():
    s = st.session_state

    st.markdown(
        """
        <div class="hero">
            <div class="hero-emoji">🤟</div>
            <h1>Znak</h1>
            <p class="tagline">Learn the sign language alphabet — gamified.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_stats()

    completed = len(s.completed_letters)
    total = len(ALPHABET)
    pct = completed / total * 100 if total else 0

    st.markdown(
        f"""
        <div class="card progress-card">
            <h3>Your progress</h3>
            <div class="progress-bar">
                <div class="progress-bar-fill" style="width:{pct}%"></div>
            </div>
            <div class="progress-meta">
                <span>{completed} of {total} letters</span>
                <span class="pct">{pct:.0f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🎯  Practice", use_container_width=True, key="qa_practice"):
            goto("practice")
    with c2:
        if st.button("⚡  Challenge", use_container_width=True, key="qa_challenge"):
            goto("challenge")
    with c3:
        if st.button("📝  Words", use_container_width=True, key="qa_words"):
            goto("word")

    if s.letter_stats:
        st.markdown("<h2 class='section-title'>📊 Per-letter accuracy</h2>", unsafe_allow_html=True)
        rows = []
        for letter in ALPHABET:
            if letter in s.letter_stats:
                stats = s.letter_stats[letter]
                total = stats["total"]
                correct = stats["correct"]
                pct = correct / total if total else 0
                rows.append((letter, correct, total, pct))

        n_cols = min(6, len(rows))
        cols = st.columns(n_cols)
        for i, (letter, correct, total, pct) in enumerate(rows):
            with cols[i % n_cols]:
                color = "#58CC02" if pct >= 0.8 else "#FFD43B" if pct >= 0.5 else "#FF5252"
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:0.5rem; background:var(--bg-card);
                                border-radius:12px; margin-bottom:0.5rem;">
                        <div style="font-size:1.3rem; font-weight:900;">{letter}</div>
                        <div style="font-size:0.8rem; color:{color}; font-weight:700;">{pct:.0%}</div>
                        <div style="font-size:0.7rem; color:var(--text-muted);">{correct}/{total}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<h2 class='section-title'>Learning path</h2>", unsafe_allow_html=True)

    for i, level in enumerate(LEVELS):
        letters_in_level = level["letters"]
        done_in_level = len([l for l in letters_in_level if l in s.completed_letters])
        unlocked = i == 0 or all(
            l in s.completed_letters for l in LEVELS[i - 1]["letters"]
        )

        st.markdown(
            f"""
            <div class="level-header">
                <div>
                    <div class="level-title">{level['name']}</div>
                    <div class="level-subtitle">{level['subtitle']}</div>
                </div>
                <div class="level-counter">{done_in_level}/{len(letters_in_level)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not unlocked:
            st.markdown(
                "<div class='locked-msg'>🔒 Finish the previous level to unlock</div>",
                unsafe_allow_html=True,
            )
            html = "<div class='letter-row'>" + "".join(
                f"<div class='letter-circle locked'>{l}</div>" for l in letters_in_level
            ) + "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            html = "<div class='letter-row'>"
            for letter in letters_in_level:
                cls = "completed" if letter in s.completed_letters else "unlocked"
                badge = "✓" if letter in s.completed_letters else letter
                html += f"<div class='letter-circle {cls}'>{badge}</div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

            cols = st.columns(len(letters_in_level))
            for col, letter in zip(cols, letters_in_level):
                with col:
                    label = "Review" if letter in s.completed_letters else f"Learn {letter}"
                    if st.button(label, key=f"learn_{letter}", use_container_width=True):
                        s.current_letter = letter
                        s.lesson_correct = 0
                        s.last_prediction = None
                        s.last_error = None
                        goto("lesson")


# ============================================================================
# LESSON
# ============================================================================

def render_lesson():
    s = st.session_state
    letter = s.current_letter

    if not letter:
        st.warning("Pick a letter on the home screen.")
        if st.button("← Home"):
            goto("home")
        return

    info = LETTERS[letter]

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("← Back", key="lesson_back"):
            goto("home")
    with c2:
        progress_pct = s.lesson_correct / REQUIRED_CORRECT * 100
        st.markdown(
            f"""
            <div class="progress-bar lesson-progress">
                <div class="progress-bar-fill" style="width:{progress_pct}%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_stats()

    left, right = st.columns(2)

    with left:
        st.markdown(
            f"""
            <div class="card letter-card">
                <div class="letter-label">Learn this letter</div>
                <div class="big-letter">{letter}</div>
                <div class="big-emoji">{info['emoji']}</div>
                <div class="letter-desc">{info['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        ctx = render_camera(target_letter=letter, key="cam_main")
        render_live_status(ctx)

        if st.button("🤟  Check pose", key="lesson_check", use_container_width=True):
            pred, err = grab_prediction(ctx)
            s.last_prediction = pred
            s.last_error = err

            if pred is not None:
                stats = s.letter_stats.setdefault(letter, {"correct": 0, "total": 0})
                stats["total"] += 1
                if _is_correct(pred, letter):
                    stats["correct"] += 1
                    s.lesson_correct += 1
                    if s.lesson_correct >= REQUIRED_CORRECT and letter not in s.completed_letters:
                        s.completed_letters = s.completed_letters | {letter}
                        s.xp += 10
                else:
                    s.lesson_correct = max(0, s.lesson_correct - 1)
            st.rerun()

    if s.last_prediction or s.last_error:
        render_feedback(s.last_prediction, target=letter, error=s.last_error)
        if s.last_prediction:
            st.markdown(
                f"<div class='lesson-counter'>Correct in a row: "
                f"<b>{s.lesson_correct}/{REQUIRED_CORRECT}</b></div>",
                unsafe_allow_html=True,
            )

    if s.lesson_correct >= REQUIRED_CORRECT:
        st.markdown(
            f"""
            <div class="lesson-complete">
                🎉 <b>Letter "{letter}" mastered!</b> +10 XP
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Home", key="lesson_home_btn", use_container_width=True):
                goto("home")
        with c2:
            if st.button("Next letter  →", key="lesson_next", use_container_width=True):
                for l in ALPHABET:
                    if l not in s.completed_letters:
                        s.current_letter = l
                        s.lesson_correct = 0
                        s.last_prediction = None
                        s.last_error = None
                        st.rerun()
                        return
                goto("home")

    st.markdown("<h3 class='section-title'>💡 Tips</h3>", unsafe_allow_html=True)
    tip_cols = st.columns(len(info["tips"]))
    for col, tip in zip(tip_cols, info["tips"]):
        with col:
            st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)


# ============================================================================
# PRACTICE
# ============================================================================

def _learned_pool():
    s = st.session_state
    return sorted(s.completed_letters) if s.completed_letters else ALPHABET


def render_practice():
    s = st.session_state

    if st.button("← Back", key="practice_back"):
        goto("home")

    st.markdown("<h1>🎯 Practice</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='tagline'>Sign the letter. No hints, no reference.</p>",
        unsafe_allow_html=True,
    )
    render_stats()

    pool = _learned_pool()
    if s.practice_letter is None:
        s.practice_letter = random.choice(pool)
        s.last_prediction = None
        s.last_error = None

    target = s.practice_letter

    show_hint = st.toggle("💡 Show hint", key="practice_hint", value=False)

    left, right = st.columns(2)
    with left:
        if show_hint:
            info = LETTERS.get(target, {"emoji": "❓", "description": "", "tips": []})
            st.markdown(
                f"""
                <div class="card letter-card">
                    <div class="letter-label">Sign this letter</div>
                    <div class="big-letter">{target}</div>
                    <div class="big-emoji">{info['emoji']}</div>
                    <div class="letter-desc">{info['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            tip_cols = st.columns(len(info["tips"])) if info["tips"] else []
            for col, tip in zip(tip_cols, info["tips"]):
                with col:
                    st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="card letter-card">
                    <div class="letter-label">Sign this letter</div>
                    <div class="big-letter">{target}</div>
                    <div class="letter-desc subtle">no hints 🤐</div>
                </div>
                <div class="practice-stats">
                    <div class="big-stat">
                        <div class="value">{s.practice_correct}</div>
                        <div class="label">Correct</div>
                    </div>
                    <div class="big-stat">
                        <div class="value">{s.practice_total}</div>
                        <div class="label">Total</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        ctx = render_camera(target_letter=target, key="cam_main")
        render_live_status(ctx)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🤟  Check", key="practice_check", use_container_width=True):
                pred, err = grab_prediction(ctx)
                s.last_prediction = pred
                s.last_error = err
                if pred is not None:
                    s.practice_total += 1
                    stats = s.letter_stats.setdefault(target, {"correct": 0, "total": 0})
                    stats["total"] += 1
                    if _is_correct(pred, target):
                        s.practice_correct += 1
                        s.xp += 2
                        stats["correct"] += 1
                        s.practice_letter = random.choice(pool)
                        s.last_prediction = None
                st.rerun()
        with c2:
            if st.button("Skip  →", key="practice_skip", use_container_width=True):
                s.practice_letter = random.choice(pool)
                s.last_prediction = None
                s.last_error = None
                st.rerun()

    if s.last_prediction or s.last_error:
        render_feedback(s.last_prediction, target=target, error=s.last_error)


# ============================================================================
# CHALLENGE
# ============================================================================

def render_challenge():
    s = st.session_state

    if st.button("← Back", key="challenge_back"):
        s.challenge_active = False
        goto("home")

    st.markdown("<h1>⚡ Challenge</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='tagline'>How many letters can you sign in 60 seconds?</p>",
        unsafe_allow_html=True,
    )

    pool = _learned_pool()

    if not s.challenge_active and s.challenge_score == 0:
        st.markdown(
            """
            <div class="card challenge-start">
                <div class="big-emoji-center">⏱️</div>
                <h2 style="text-align:center;">Ready?</h2>
                <p style="text-align:center; color:var(--text-secondary);">
                    60 seconds. Random letters.<br>
                    +5 XP per correct sign.<br>
                    A wrong sign costs ❤️ — three strikes ends the run.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀  Start", key="challenge_start", use_container_width=True):
            s.challenge_active = True
            s.challenge_score = 0
            s.challenge_mistakes = 0
            s.challenge_start_ts = time.time()
            s.challenge_letter = random.choice(pool)
            s.last_prediction = None
            s.last_error = None
            st.rerun()
        return

    elapsed = time.time() - s.challenge_start_ts
    time_left = max(0.0, CHALLENGE_DURATION - elapsed)
    is_over = (not s.challenge_active) or time_left <= 0 or s.challenge_mistakes >= 3

    if is_over:
        s.challenge_active = False
        st.markdown(
            f"""
            <div class="card challenge-end">
                <div class="big-emoji-center">🏁</div>
                <h2 style="text-align:center;">Done!</h2>
                <div class="challenge-final">
                    <div class="big-stat">
                        <div class="value">{s.challenge_score}</div>
                        <div class="label">Letters</div>
                    </div>
                    <div class="big-stat">
                        <div class="value">+{s.challenge_score * 5}</div>
                        <div class="label">XP earned</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔁  Try again", key="challenge_retry", use_container_width=True):
                s.challenge_score = 0
                s.challenge_mistakes = 0
                st.rerun()
        with c2:
            if st.button("Home", key="challenge_home", use_container_width=True):
                s.challenge_score = 0
                s.challenge_mistakes = 0
                goto("home")
        return

    target = s.challenge_letter
    time_pct = time_left / CHALLENGE_DURATION * 100

    st.markdown(
        f"""
        <div class="challenge-hud">
            <div class="hud-cell"><span>⏱️</span><b>{time_left:.0f}s</b></div>
            <div class="hud-cell"><span>💎</span><b>{s.challenge_score}</b></div>
            <div class="hud-cell hud-hearts">
                {'❤️' * (3 - s.challenge_mistakes)}{'🖤' * s.challenge_mistakes}
            </div>
        </div>
        <div class="progress-bar timer-bar">
            <div class="progress-bar-fill timer-fill" style="width:{time_pct}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="card letter-card">
                <div class="letter-label">Sign</div>
                <div class="big-letter">{target}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        ctx = render_camera(target_letter=target, key="cam_main")
        render_live_status(ctx)
        if st.button("🤟  Check", key="challenge_check", use_container_width=True):
            pred, err = grab_prediction(ctx)
            s.last_prediction = pred
            s.last_error = err
            if pred is not None:
                stats = s.letter_stats.setdefault(target, {"correct": 0, "total": 0})
                stats["total"] += 1
                if _is_correct(pred, target):
                    s.challenge_score += 1
                    s.xp += 5
                    stats["correct"] += 1
                    s.challenge_letter = random.choice(pool)
                    s.last_prediction = None
                else:
                    s.challenge_mistakes += 1
            st.rerun()

    if s.last_prediction or s.last_error:
        render_feedback(s.last_prediction, target=target, error=s.last_error)


# ============================================================================
# WORDS
# ============================================================================

def render_word_mode():
    s = st.session_state

    if st.button("← Back", key="word_back"):
        goto("home")

    st.markdown("<h1>📝 Words</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline'>Sign the word, letter by letter.</p>", unsafe_allow_html=True)
    render_stats()

    if s.word_target is None:
        st.markdown("<h3 class='section-title'>Pick difficulty</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        diffs = [
            ("easy", "🟢 Easy", "3 letters"),
            ("medium", "🟡 Medium", "4-5 letters"),
            ("hard", "🔴 Hard", "6+ letters"),
        ]
        for col, (key, label, sub) in zip([c1, c2, c3], diffs):
            with col:
                if st.button(f"{label}\n{sub}", key=f"diff_{key}", use_container_width=True):
                    s.word_target = random.choice(WORDS_BY_DIFFICULTY[key])
                    s.word_index = 0
                    s.word_done = False
                    s.last_prediction = None
                    s.last_error = None
                    st.rerun()
        return

    word = s.word_target
    idx = s.word_index

    word_html = "<div class='word-display'>"
    for i, ch in enumerate(word):
        if ch == " ":
            word_html += "<span class='space'> </span>"
            continue
        if i < idx:
            word_html += f"<span class='done'>{ch}</span>"
        elif i == idx and not s.word_done:
            word_html += f"<span class='current'>{ch}</span>"
        else:
            word_html += f"<span class='pending'>{ch}</span>"
    word_html += "</div>"
    st.markdown(word_html, unsafe_allow_html=True)

    if s.word_done:
        st.markdown(
            f"""
            <div class="lesson-complete">
                🎉 <b>"{word}" — well done!</b> +{len(word) * 3} XP
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔁  New word", key="word_more", use_container_width=True):
                s.word_target = None
                st.rerun()
        with c2:
            if st.button("Home", key="word_home", use_container_width=True):
                s.word_target = None
                goto("home")
        return

    target = word[idx] if idx < len(word) else None
    if target == " ":
        s.word_index += 1
        st.rerun()

    left, right = st.columns(2)
    with left:
        info = LETTERS.get(target, {"emoji": "❓"})
        st.markdown(
            f"""
            <div class="card letter-card">
                <div class="letter-label">Sign this letter</div>
                <div class="big-letter">{target}</div>
                <div class="big-emoji">{info['emoji']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        ctx = render_camera(target_letter=target, key="cam_main")
        render_live_status(ctx)
        if st.button("🤟  Check", key="word_check", use_container_width=True):
            pred, err = grab_prediction(ctx)
            s.last_prediction = pred
            s.last_error = err
            if pred is not None and _is_correct(pred, target):
                s.word_index += 1
                s.last_prediction = None
                if s.word_index >= len(word):
                    s.word_done = True
                    s.xp += len(word) * 3
            st.rerun()

    if s.last_prediction or s.last_error:
        render_feedback(s.last_prediction, target=target, error=s.last_error)
