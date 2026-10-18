import pickle
import string
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="🧠",
    layout="wide",
)


# -----------------------------
# Claymorphism styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #0d1117;
        color: #f2f4f8;
    }

    .block-container {
        max-width: 1180px;
        padding: 4.5rem 2rem 3rem;
    }

    .main-title {
        color: #f5f7fb;
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        font-weight: 800;
        text-align: left;
        letter-spacing: -0.03em;
        margin-bottom: 0.7rem;
    }

    .subtitle {
        color: #b4bdca;
        text-align: left;
        font-size: 1.1rem;
        margin-bottom: 2.4rem;
    }

    h2, h3 {
        color: #f2f4f8 !important;
    }

    label {
        color: #d2d8e1 !important;
        font-weight: 600 !important;
    }

    textarea {
        border: 1px solid #4a5361 !important;
        border-radius: 12px !important;
        background: #222731 !important;
        color: #f2f4f8 !important;
        min-height: 78px !important;
    }

    textarea:focus {
        border: 1px solid #e0525d !important;
        box-shadow: 0 0 0 2px rgba(224, 82, 93, 0.18) !important;
    }

    div.stButton > button {
        width: 100%;
        min-height: 46px;
        border: 1px solid #48515e;
        border-radius: 10px;
        background: #171c24;
        color: #f2f4f8;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #252c36;
        border-color: #7c8795;
        color: #ffffff;
    }

    .result-box {
        margin-top: 1.5rem;
        padding: 1.05rem 1.25rem;
        border-radius: 10px;
        background: #123c2d;
        color: #6ce49a;
        line-height: 1.7;
        overflow-wrap: anywhere;
        font-size: 1.05rem;
    }

    .action-caption {
        color: #8e99a8;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.25rem 0 0.5rem;
    }

    [data-testid="stExpander"] {
        border-color: #303844;
        color: #cbd2dc;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent;
        border: 0;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 2.5rem 1rem 2rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Load model files
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, compile=False)

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    with open(MAX_LEN_PATH, "rb") as file:
        max_len = int(pickle.load(file))

    return model, tokenizer, max_len


try:
    model, tokenizer, max_len = load_resources()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()


# -----------------------------
# Prediction functions
# -----------------------------
def clean_text(text):
    return text.lower().translate(
        str.maketrans("", "", string.punctuation)
    ).strip()


def predictor(text):
    text = clean_text(text)

    if not text:
        return "<UNKNOWN>"

    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return "<UNKNOWN>"

    padded_sequence = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="pre",
    )

    prediction = model.predict(padded_sequence, verbose=0)
    predicted_index = int(np.argmax(prediction[0]))

    return tokenizer.index_word.get(predicted_index, "<UNKNOWN>")


def generate_quote(seed_text, num_words):
    generated_text = clean_text(seed_text)

    if not generated_text:
        return ""

    for _ in range(num_words):
        sequence = tokenizer.texts_to_sequences([generated_text])[0]

        if not sequence:
            break

        padded_sequence = pad_sequences(
            [sequence],
            maxlen=max_len,
            padding="pre",
        )

        prediction = model.predict(padded_sequence, verbose=0)
        predicted_index = int(np.argmax(prediction[0]))

        predicted_word = tokenizer.index_word.get(predicted_index)

        if not predicted_word:
            break

        generated_text += " " + predicted_word

    return generated_text


# -----------------------------
# User interface
# -----------------------------
st.markdown(
    '<div class="main-title">🧠 Next Word Prediction (LSTM)</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Enter a sentence and the model will predict the next word.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Your prompt")

    prompt = st.text_area(
        "Enter text",
        placeholder="Example: the meaning of life",
        height=92,
        key="prompt",
        label_visibility="collapsed",
    )

    action_column, generate_column, length_column = st.columns(
        [1, 1, 1.15],
        gap="medium",
    )

    with action_column:
        st.markdown('<div class="action-caption">Single word</div>', unsafe_allow_html=True)
        predict_clicked = st.button(
            "Predict next word",
            key="predict_button",
            use_container_width=True,
        )

    with generate_column:
        st.markdown('<div class="action-caption">Continue writing</div>', unsafe_allow_html=True)
        generate_clicked = st.button(
            "Generate sentence",
            key="generate_button",
            use_container_width=True,
        )

    with length_column:
        st.markdown('<div class="action-caption">Generated words</div>', unsafe_allow_html=True)
        number_of_words = st.slider(
            "Number of words",
            min_value=1,
            max_value=50,
            value=10,
            label_visibility="collapsed",
        )

    if predict_clicked or generate_clicked:
        if not prompt.strip():
            st.warning("Enter a prompt before selecting an action.")
        elif predict_clicked:
            next_word = predictor(prompt)
            st.markdown(
                f"""
                <div class="result-box">
                    <strong>Predicted next word</strong><br>
                    {next_word}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            generated_text = generate_quote(prompt, number_of_words)
            st.markdown(
                f"""
                <div class="result-box">
                    <strong>Generated text</strong><br>
                    {generated_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.expander("Model information"):
    st.write(f"Maximum sequence length: `{max_len}`")
    st.write(f"Input shape: `{model.input_shape}`")
    st.write(f"Output shape: `{model.output_shape}`")