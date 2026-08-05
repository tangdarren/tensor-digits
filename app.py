"""Streamlit entry point for TensorDigits."""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.insights import (
    artifacts_available,
    build_insight_bundle,
    figure_confusion_matrix,
    figure_misclassified_examples,
    figure_training_curves,
    load_insight_sources,
)
from src.preprocessing import BlankDrawingError, InvalidDrawingError
from src.prediction import PredictionResult, get_model, predict_digit
from src.training import EVALUATION_PATH, HISTORY_PATH, MODEL_PATH

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

        .td-predict-panel {
            border: 2px solid #000000;
            background: #FFFFFF;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1.25rem 1.25rem 1rem 1.25rem;
            margin-bottom: 1rem;
            text-align: center;
        }

        .td-digit {
            font-size: 5.5rem;
            font-weight: 700;
            line-height: 1;
            letter-spacing: -0.04em;
            color: #000000;
            margin: 0;
        }

        .td-digit-placeholder {
            font-size: 2.5rem;
            font-weight: 700;
            color: #BBBBBB;
            margin: 0;
        }

        .td-panel {
            border: 2px solid #000000;
            background: #F5F5F5;
            min-height: 72px;
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

        .td-error {
            border: 2px solid #000000;
            background: #FFFFFF;
            padding: 0.85rem 1rem;
            margin: 0 0 1rem 0;
            font-size: 0.95rem;
            color: #000000;
            text-align: center;
        }

        .td-prob-table {
            width: 100%;
            border: 2px solid #000000;
            border-collapse: collapse;
            margin-bottom: 0.5rem;
            background: #FFFFFF;
        }

        .td-prob-table td {
            padding: 0.4rem 0.65rem;
            border-bottom: 1px solid #DDDDDD;
            vertical-align: middle;
            font-size: 0.9rem;
            color: #000000;
        }

        .td-prob-table tr:last-child td {
            border-bottom: none;
        }

        .td-prob-table tr.td-top td {
            font-weight: 700;
            background: #F5F5F5;
        }

        .td-digit-cell {
            width: 1.75rem;
            font-variant-numeric: tabular-nums;
        }

        .td-bar-track {
            height: 8px;
            background: #E8E8E8;
            width: 100%;
        }

        .td-bar-fill {
            height: 8px;
            background: #000000;
        }

        .td-pct-cell {
            width: 3.5rem;
            text-align: right;
            font-variant-numeric: tabular-nums;
            white-space: nowrap;
        }

        .td-insights-summary {
            font-size: 0.95rem;
            color: #333333;
            margin: 0 0 1rem 0;
            line-height: 1.5;
        }

        div[data-testid="stExpander"] {
            border: 2px solid #000000;
            background: #FFFFFF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_cached_model():
    """Load the classifier once per process; reuse across reruns."""
    if not MODEL_PATH.exists():
        return None
    return get_model(MODEL_PATH)


@st.cache_resource(show_spinner="Preparing model insights…")
def load_cached_insights():
    """Compute evaluation insights once per process."""
    if not artifacts_available():
        return None
    return build_insight_bundle()


def render_model_insights() -> None:
    """Collapsed insights panel kept secondary to the drawing UI."""
    with st.expander("Model insights", expanded=False):
        if not (HISTORY_PATH.exists() and EVALUATION_PATH.exists()):
            st.markdown(
                '<p class="td-insights-summary">'
                "Training artifacts are missing. Run <code>python train.py</code> "
                "to generate history and evaluation metrics."
                "</p>",
                unsafe_allow_html=True,
            )
            return

        try:
            history, evaluation = load_insight_sources()
        except Exception as exc:  # noqa: BLE001
            st.markdown(
                f'<div class="td-error">Could not load training artifacts: {exc}</div>',
                unsafe_allow_html=True,
            )
            return

        accuracy = float(evaluation.get("test_accuracy", 0.0)) * 100.0
        loss = float(evaluation.get("test_loss", 0.0))
        st.markdown(
            f'<p class="td-insights-summary">'
            f"Test accuracy <strong>{accuracy:.2f}%</strong> · "
            f"test loss <strong>{loss:.4f}</strong> · "
            f"{int(evaluation.get('num_test_samples', 0)):,} MNIST test images"
            f"</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="td-section-label">Training curves</p>', unsafe_allow_html=True)
        fig_curves = figure_training_curves(history)
        st.pyplot(fig_curves, clear_figure=True)
        plt_close(fig_curves)

        st.markdown('<p class="td-section-label">Test-set analysis</p>', unsafe_allow_html=True)
        if not MODEL_PATH.exists():
            st.markdown(
                '<p class="td-insights-summary">'
                "Saved model not found, so the confusion matrix and error gallery "
                "cannot be generated."
                "</p>",
                unsafe_allow_html=True,
            )
            return

        # Defer the MNIST re-evaluation until requested so drawing stays snappy.
        if st.button("Load confusion matrix & errors", use_container_width=True):
            st.session_state.show_deep_insights = True

        if not st.session_state.get("show_deep_insights"):
            st.markdown(
                '<p class="td-insights-summary">'
                "Training curves load instantly from saved history. Click above to "
                "evaluate the test set for a confusion matrix and sample mistakes."
                "</p>",
                unsafe_allow_html=True,
            )
            return

        try:
            bundle = load_cached_insights()
        except Exception as exc:  # noqa: BLE001
            st.markdown(
                f'<div class="td-error">Could not compute model insights: {exc}</div>',
                unsafe_allow_html=True,
            )
            return

        if bundle is None:
            st.markdown(
                '<p class="td-insights-summary">Model insights are unavailable.</p>',
                unsafe_allow_html=True,
            )
            return

        st.markdown('<p class="td-section-label">Confusion matrix</p>', unsafe_allow_html=True)
        fig_cm = figure_confusion_matrix(bundle.confusion_matrix)
        st.pyplot(fig_cm, clear_figure=True)
        plt_close(fig_cm)

        st.markdown(
            '<p class="td-section-label">Incorrect predictions</p>',
            unsafe_allow_html=True,
        )
        fig_errors = figure_misclassified_examples(
            bundle.error_images,
            bundle.error_true,
            bundle.error_pred,
        )
        if fig_errors is None:
            st.markdown(
                '<p class="td-insights-summary">No misclassified test examples found.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.pyplot(fig_errors, clear_figure=True)
            plt_close(fig_errors)


def plt_close(fig) -> None:
    """Close a Matplotlib figure to avoid memory growth across Streamlit reruns."""
    plt.close(fig)


def render_probability_breakdown(result: PredictionResult) -> None:
    """Render a compact black-and-white probability table for digits 0–9."""
    rows: list[str] = []
    for digit, probability in enumerate(result.probabilities):
        pct = float(probability) * 100.0
        width = max(0.0, min(100.0, pct))
        top_class = "td-top" if digit == result.digit else ""
        rows.append(
            "<tr class='"
            f"{top_class}'>"
            f"<td class='td-digit-cell'>{digit}</td>"
            "<td>"
            "<div class='td-bar-track'>"
            f"<div class='td-bar-fill' style='width:{width:.2f}%'></div>"
            "</div>"
            "</td>"
            f"<td class='td-pct-cell'>{pct:.1f}%</td>"
            "</tr>"
        )
    st.markdown(
        "<table class='td-prob-table'>" + "".join(rows) + "</table>",
        unsafe_allow_html=True,
    )


if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

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

model = None
model_load_error = None
try:
    model = load_cached_model()
    if model is None:
        model_load_error = (
            f"No trained model found at `{MODEL_PATH.name}`. "
            "Run `python train.py` to create one, then refresh this page."
        )
except Exception as exc:  # noqa: BLE001
    model_load_error = f"Could not load the trained model: {exc}"

if model_load_error:
    st.markdown(f'<div class="td-error">{model_load_error}</div>', unsafe_allow_html=True)

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
        disabled=not has_strokes or model is None,
    )

if clear_clicked:
    st.session_state.canvas_version += 1
    st.session_state.prediction = None
    st.session_state.prediction_error = None
    st.rerun()

if predict_clicked:
    st.session_state.prediction = None
    st.session_state.prediction_error = None

    image_data = None if canvas_result is None else canvas_result.image_data
    if image_data is None:
        st.session_state.prediction_error = (
            "No drawing data was captured. Try drawing the digit again."
        )
    elif model is None:
        st.session_state.prediction_error = model_load_error or "Model is unavailable."
    else:
        try:
            st.session_state.prediction = predict_digit(image_data, model)
        except BlankDrawingError as exc:
            st.session_state.prediction_error = str(exc)
        except InvalidDrawingError as exc:
            st.session_state.prediction_error = str(exc)
        except RuntimeError as exc:
            st.session_state.prediction_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            st.session_state.prediction_error = f"Unexpected prediction error: {exc}"

prediction: PredictionResult | None = st.session_state.prediction
error_message: str | None = st.session_state.prediction_error

if error_message:
    st.markdown(f'<div class="td-error">{error_message}</div>', unsafe_allow_html=True)

st.markdown('<p class="td-section-label">Prediction</p>', unsafe_allow_html=True)
if prediction is not None:
    st.markdown(
        f'<div class="td-predict-panel"><p class="td-digit">{prediction.digit}</p></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="td-predict-panel">'
        '<p class="td-digit-placeholder">—</p>'
        '<p class="td-panel-hint">Predicted digit will appear here</p>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown('<p class="td-section-label">Confidence</p>', unsafe_allow_html=True)
if prediction is not None:
    confidence_pct = prediction.confidence * 100.0
    st.markdown(
        f'<div class="td-panel">'
        f'<p class="td-panel-value">{confidence_pct:.1f}%</p>'
        f'<p class="td-panel-hint">Confidence for digit {prediction.digit}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="td-panel">'
        '<p class="td-panel-value">—</p>'
        '<p class="td-panel-hint">Confidence score will appear here</p>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown('<p class="td-section-label">All digit probabilities</p>', unsafe_allow_html=True)
if prediction is not None:
    render_probability_breakdown(prediction)
else:
    st.markdown(
        '<div class="td-panel">'
        '<p class="td-panel-hint">Probabilities for 0–9 will appear here</p>'
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)
render_model_insights()