"""Streamlit entry point for TensorDigits."""

from __future__ import annotations

import streamlit as st
from streamlit_drawable_canvas import st_canvas

CANVAS_SIZE = 280
STROKE_WIDTH = 18

st.set_page_config(
    page_title="TensorDigits",
    page_icon="⬛",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            max-width: 640px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .td-title {
            font-size: 2.75rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            text-align: center;
            margin: 0 0 0.5rem 0;
            color: #000000;
        }

        .td-subtitle {
            font-size: 1.05rem;
            line-height: 1.55;
            text-align: center;
            color: #333333;
            margin: 0 0 0.75rem 0;
        }

        .td-instructions {
            font-size: 0.95rem;
            line-height: 1.5;
            text-align: center;
            color: #555555;
            margin: 0 0 1.75rem 0;
        }

        .td-section-label {
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #666666;
            margin: 0 0 0.6rem 0;
        }

        /* Center and frame the drawable-canvas iframe. */
        div[data-testid="stIFrame"] {
            display: flex;
            justify-content: center;
            margin-bottom: 0.85rem;
        }

        iframe[title="streamlit_drawable_canvas.st_canvas"] {
            border: 2px solid #000000 !important;
            display: block;
            margin: 0 auto;
        }

        div[data-testid="stHorizontalBlock"] button {
            border: 2px solid #000000 !important;
            border-radius: 0 !important;
            font-weight: 600 !important;
        }

        .td-panel {
            border: 2px solid #000000;
            background: #F5F5F5;
            min-height: 88px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            text-align: center;
        }

        .td-panel-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #000000;
            line-height: 1.2;
            margin: 0;
        }

        .td-panel-hint {
            font-size: 0.9rem;
            color: #666666;
            margin: 0.35rem 0 0 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "submit_message" not in st.session_state:
    st.session_state.submit_message = None

st.markdown('<h1 class="td-title">TensorDigits</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="td-subtitle">'
    "Draw a digit from 0 through 9. The model will predict which number you wrote."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="td-instructions">'
    "Use your mouse or trackpad to draw one digit in white on the black canvas, "
    "then click Predict. Use Clear to start over."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown('<p class="td-section-label">Drawing canvas</p>', unsafe_allow_html=True)

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=STROKE_WIDTH,
    stroke_color="#FFFFFF",
    background_color="#000000",
    update_streamlit=True,
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    drawing_mode="freedraw",
    display_toolbar=False,
    key=f"digit_canvas_{st.session_state.canvas_version}",
)

has_strokes = bool(
    canvas_result is not None
    and canvas_result.json_data is not None
    and len(canvas_result.json_data.get("objects", [])) > 0
)

clear_col, predict_col = st.columns(2)
with clear_col:
    clear_clicked = st.button("Clear", use_container_width=True)
with predict_col:
    predict_clicked = st.button(
        "Predict",
        type="primary",
        use_container_width=True,
        disabled=not has_strokes,
    )

if clear_clicked:
    st.session_state.canvas_version += 1
    st.session_state.submit_message = None
    st.rerun()

if predict_clicked:
    # Model wiring comes in a later step; keep the submit path ready.
    st.session_state.submit_message = (
        "Drawing captured. Prediction will appear here once the model is connected."
    )

st.markdown('<p class="td-section-label">Prediction</p>', unsafe_allow_html=True)
if st.session_state.submit_message:
    st.markdown(
        f'<div class="td-panel">'
        f'<p class="td-panel-value">—</p>'
        f'<p class="td-panel-hint">{st.session_state.submit_message}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="td-panel">'
        '<p class="td-panel-value">—</p>'
        '<p class="td-panel-hint">Predicted digit will appear here</p>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown('<p class="td-section-label">Confidence</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="td-panel">'
    '<p class="td-panel-value">—</p>'
    '<p class="td-panel-hint">Confidence score will appear here</p>'
    "</div>",
    unsafe_allow_html=True,
)
