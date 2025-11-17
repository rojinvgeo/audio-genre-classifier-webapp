import streamlit as st
import librosa
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(
    page_title="Music Mood Classifier",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="auto",
)

# ------------------------------------
# CUSTOM CSS (BEAUTIFUL DESIGN)
# ------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #1c1f4a, #2a2d62);
    color: white !important;
}

h1, h2, h3, h4, h5, h6, p, span {
    color: white !important;
}

.upload-box {
    padding: 30px;
    border-radius: 18px;
    background-color: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    margin-top: 20px;
    background-color: rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------
# LOAD MODEL + SCALER
# ------------------------------------
MODEL_PATH = "models/music_genre_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------------------
# FEATURE EXTRACTION (Your Matching Extractor)
# ------------------------------------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

        return np.hstack([mfccs, chroma, mel, contrast, tonnetz]).reshape(1, -1)

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# ------------------------------------
# MOOD MAPPING
# ------------------------------------
GENRE_TO_MOOD = {
    "classical": "😌 Calm / Relaxed",
    "jazz": "🎷 Chill / Smooth",
    "blues": "😔 Sad / Emotional",
    "hiphop": "💪 Energetic / Confident",
    "rock": "🔥 Powerful / Aggressive",
    "metal": "🤘 Intense / Angry",
    "reggae": "🌴 Chill / Happy",
    "pop": "😊 Happy / Bright",
    "country": "🏞 Relaxed / Storytelling",
    "disco": "💃 Party / Dance"
}

# ------------------------------------
# HEADER
# ------------------------------------
st.markdown("<h1 style='text-align:center;'>🎵 Music Genre & Mood Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload an audio file to detect its genre and mood instantly. 🎧✨</p>", unsafe_allow_html=True)

# ------------------------------------
# UPLOAD BOX
# ------------------------------------
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file)
    with st.spinner("🎶 Analyzing your music..."):
        
        features = extract_features(uploaded_file)
        scaled = scaler.transform(features)
        predicted_genre = model.predict(scaled)[0]
        predicted_mood = GENRE_TO_MOOD.get(predicted_genre, "🙂 Neutral Mood")

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------
    # RESULT CARD
    # ------------------------------------
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center;'>🎧 Prediction Results</h2>", unsafe_allow_html=True)

    st.markdown(f"<h3>🎼 Genre: <b>{predicted_genre.capitalize()}</b></h3>", unsafe_allow_html=True)
    st.markdown(f"<h3>🧠 Mood: <b>{predicted_mood}</b></h3>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("</div>", unsafe_allow_html=True)
