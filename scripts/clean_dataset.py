import os
import librosa
import hashlib

DATASET_DIR = "data/genres"
CLEAN_LIST_PATH = "data/clean_files.txt"

def get_file_hash(path):
    """Return SHA1 hash of audio file (for duplicate detection)."""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def is_corrupted(path):
    """Try loading the file — if fails, mark as corrupted."""
    try:
        audio, sr = librosa.load(path, duration=3)  # load only 3 sec to test
        return False
    except:
        return True


def has_valid_duration(path, min_sec=25, max_sec=35):
    """Check if audio duration is close to 30 seconds."""
    try:
        duration = librosa.get_duration(filename=path)
        return min_sec <= duration <= max_sec
    except:
        return False


def is_silent(path, threshold=0.001):
    """Check if file has almost no sound."""
    try:
        audio, sr = librosa.load(path)
        if audio.mean() == 0:
            return True
        return audio.max() < threshold
    except:
        return True


def clean_dataset():
    print("🔍 Cleaning dataset...\n")

    file_hashes = {}
    clean_files = []

    for genre in os.listdir(DATASET_DIR):
        genre_path = os.path.join(DATASET_DIR, genre)

        for filename in os.listdir(genre_path):
            file_path = os.path.join(genre_path, filename)

            print(f"📌 Checking: {file_path}")

            # Skip non-wav files
            if not filename.endswith(".wav"):
                continue

            # 1️⃣ Corrupted?
            if is_corrupted(file_path):
                print(f"❌ Corrupted → removed: {filename}")
                continue

            # 2️⃣ Silent?
            if is_silent(file_path):
                print(f"❌ Silent → removed: {filename}")
                continue

            # 3️⃣ Wrong duration?
            if not has_valid_duration(file_path):
                print(f"❌ Wrong duration → removed: {filename}")
                continue

            # 4️⃣ Duplicate?
            file_hash = get_file_hash(file_path)
            if file_hash in file_hashes:
                print(f"❌ Duplicate of {file_hashes[file_hash]} → removed: {filename}")
                continue
            else:
                file_hashes[file_hash] = file_path

            # If passed all checks
            clean_files.append(file_path)

    # Save clean file list
    with open(CLEAN_LIST_PATH, "w") as f:
        for fpath in clean_files:
            f.write(fpath + "\n")

    print("\n🎉 Cleaning finished!")
    print(f"✅ Total clean audio files: {len(clean_files)}")
    print(f"📄 Saved list to: {CLEAN_LIST_PATH}")


if __name__ == "__main__":
    clean_dataset()
