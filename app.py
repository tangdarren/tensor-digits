"""Streamlit entry point for TensorDigits."""

import streamlit as st

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
            padding-top: 3rem;
            padding-bottom: 3rem;
        }

        .td-title {
            font-size: 2.75rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            text-align: center;
            margin: 0 0 0.75rem 0;
            color: #000000;
        }

        .td-subtitle {
            font-size: 1.05rem;
            line-height: 1.55;
            text-align: center;
            color: #333333;
            margin: 0 0 2.5rem 0;
        }

        .td-panel {
            border: 2px solid #000000;
            background: #FFFFFF;
            min-height: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.25rem;
        }

        .td-panel-label {
            font-size: 0.95rem;
            color: #666666;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .td-result {
            border: 2px solid #000000;
            background: #F5F5F5;
            min-height: 96px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .td-result-label {
            font-size: 0.95rem;
            color: #666666;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="td-title">TensorDigits</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="td-subtitle">'
    "Draw a digit from 0 through 9. The model will predict which number you wrote."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="td-panel"><span class="td-panel-label">Drawing canvas — coming soon</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="td-result"><span class="td-result-label">Prediction — coming soon</span></div>',
    unsafe_allow_html=True,
)
