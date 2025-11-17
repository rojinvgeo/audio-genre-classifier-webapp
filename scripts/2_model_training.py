import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ----------------------------
# Load dataset
# ----------------------------
DATASET_PATH = "features/features.csv"
df = pd.read_csv(DATASET_PATH)

print("📂 Loaded dataset:", df.shape)

# ----------------------------
# Split features & labels
# ----------------------------
X = df.drop("label", axis=1)
y = df["label"]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("📊 Train shape:", X_train.shape)
print("📊 Test shape:", X_test.shape)

# ----------------------------
# Scale features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Classifier
# ----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)

print("🚀 Training model...")
model.fit(X_train_scaled, y_train)

# ----------------------------
# Evaluate the model
# ----------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy:.4f}")

print("\n📘 Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save model + scaler
# ----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/music_genre_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n💾 Model saved to: models/music_genre_model.pkl")
print("💾 Scaler saved to: models/scaler.pkl")
