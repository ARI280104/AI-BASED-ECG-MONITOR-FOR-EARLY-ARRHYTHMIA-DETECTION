import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from feature_extraction import FEATURE_COLUMNS
DATASET_PATH = "ecg_dataset.csv"
MODEL_PATH   = "models/model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"
def train():
    print("================================================")
    print("  CardioSense AI — Training Ensemble Model")
    print("================================================")
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {len(df)} rows, {df['label'].nunique()} classes")
    # ── Drop classes with too few samples to split train/test ────
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= 5].index
    df = df[df["label"].isin(valid_labels)]
    print(f"After filtering rare classes: {len(df)} rows, "
          f"{df['label'].nunique()} classes remain")
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=15, random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss', verbosity=0
    )
    svm = SVC(kernel='rbf', C=1.0, probability=True)
    ensemble = VotingClassifier(
        estimators=[('random_forest', rf), ('xgboost', xgb), ('svm', svm)],
        voting='soft'
    )
    print("\nTraining... (this may take a few minutes)")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        zero_division=0
    ))
    # ── Save everything ─────────────────────────────────────────────
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(ensemble, MODEL_PATH)
    joblib.dump(scaler,   SCALER_PATH)
    joblib.dump(encoder,  ENCODER_PATH)

    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved scaler  -> {SCALER_PATH}")
    print(f"Saved encoder -> {ENCODER_PATH}")
    print("\nTraining complete. You can now run main.py to use this model.")

    return ensemble, scaler, encoder


if __name__ == "__main__":
    train()
