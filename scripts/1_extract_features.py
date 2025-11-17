import os
import librosa
import numpy as np
import pandas as pd

# --------------------------------
# USE CLEAN FILE LIST
# --------------------------------
CLEAN_LIST = "data/clean_files.txt"
OUTPUT_PATH = "features/features.csv"

# Create output folder
os.makedirs("features", exist_ok=True)

# --------------------------------
# FEATURE EXTRACTION
# --------------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, duration=30)

        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)

        return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

    except Exception as e:
        print(f"❌ Error: {file_path} → {e}")
        return None


# --------------------------------
# LOAD CLEANED FILES
# --------------------------------
print("\n🎵 Extracting audio features...")

with open(CLEAN_LIST, "r") as f:
    file_list = [line.strip() for line in f.readlines()]

rows = []
total = 0

# --------------------------------
# LOOP THROUGH CLEAN FILES
# --------------------------------
for file_path in file_list:

    # genre = the folder name = genre label
    genre = file_path.split("\\")[-2]  # Windows path → genre between folders

    features = extract_features(file_path)

    if features is not None:
        rows.append([genre] + list(features))
        total += 1
        print(f"✅ Done: {file_path}")

# --------------------------------
# SAVE DATA
# --------------------------------
columns = ["label"] + [f"f{i}" for i in range(len(rows[0]) - 1)]
df = pd.DataFrame(rows, columns=columns)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\n🎉 Features saved to: {OUTPUT_PATH}")
print(f"🎧 Total processed tracks: {total}")
