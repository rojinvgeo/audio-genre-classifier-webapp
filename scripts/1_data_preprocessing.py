
import os
import librosa
import numpy as np
import pandas as pd

DATA_PATH = "../data/"
FEATURES_PATH = "../features/features.csv"

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        features = np.hstack([mfcc, chroma, contrast])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_features_dataset():
    data = []
    for genre in os.listdir(DATA_PATH):
        genre_folder = os.path.join(DATA_PATH, genre)
        if not os.path.isdir(genre_folder):
            continue
        for file in os.listdir(genre_folder):
            file_path = os.path.join(genre_folder, file)
            features = extract_features(file_path)
            if features is not None:
                data.append([genre] + list(features))
    df = pd.DataFrame(data)
    df.to_csv(FEATURES_PATH, index=False)
    print("✅ Features saved to:", FEATURES_PATH)

if __name__ == "__main__":
    create_features_dataset()
