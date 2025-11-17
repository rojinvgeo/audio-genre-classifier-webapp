import librosa
import numpy as np
import joblib
import sys

# -------------------------------
# PATHS
# -------------------------------
MODEL_PATH = "models/music_genre_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# -------------------------------
# FEATURE EXTRACTION (MATCHES TRAINING)
# -------------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)

        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)

        # Final feature vector (173 features)
        return np.hstack([mfccs, chroma, mel, contrast, tonnetz]).reshape(1, -1)

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# -------------------------------
# PREDICT GENRE
# -------------------------------
def predict_genre(audio_file):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(audio_file)
    if features is None:
        return "Error extracting features"

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    return prediction

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🎵 Usage: python scripts/3_predict.py <path_to_audio>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print("🎵 Predicting genre...")
    genre = predict_genre(audio_path)
    print(f"🎶 Predicted Genre: {genre}")
