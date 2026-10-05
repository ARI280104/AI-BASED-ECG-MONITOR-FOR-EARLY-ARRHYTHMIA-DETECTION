import os
import numpy as np
import joblib
from feature_extraction import FEATURE_COLUMNS, features_to_vector
MODEL_PATH   = "models/model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"
HIGH_RISK_CLASSES = {
    "Ventricular Fibrillation",
    "Ventricular Tachycardia",
    "ST Elevation (STEMI)",
    "Third Degree Heart Block",
}
class Classifier:
    def __init__(self):
        self.model   = None
        self.scaler  = None
        self.encoder = None
        self.ready   = False
        self._try_load()
    def _try_load(self):
        if (os.path.exists(MODEL_PATH) and
            os.path.exists(SCALER_PATH) and
            os.path.exists(ENCODER_PATH)):
            self.model   = joblib.load(MODEL_PATH)
            self.scaler  = joblib.load(SCALER_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            self.ready   = True
            print("[classifier] Trained ML model loaded successfully.")
        else:
            self.ready = False
            print("[classifier] No trained model found — using rule-based "
                  "fallback. Run build_dataset.py then train_model.py to "
                  "enable the real ML model.")
    def classify(self, features: dict):
        if self.ready:
            return self._classify_ml(features)
        return self._classify_rule_based(features)
    #  Real ML model path 
    def _classify_ml(self, features: dict):
        vec = features_to_vector(features).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)
        pred_idx = self.model.predict(vec_scaled)[0]
        probs    = self.model.predict_proba(vec_scaled)[0]
        predicted_class = self.encoder.inverse_transform([pred_idx])[0]
        confidence = round(float(np.max(probs)) * 100, 2)
        all_probs = {
            self.encoder.inverse_transform([i])[0]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }
        risk = ("HIGH" if predicted_class in HIGH_RISK_CLASSES else
                "LOW"  if predicted_class == "Normal Sinus Rhythm" else
                "MEDIUM")
        return {
            "prediction": predicted_class,
            "confidence_pct": confidence,
            "risk_level": risk,
            "all_probs": all_probs,
            "model_type": "ml_ensemble",
        }
    #  Fallback used before training is done 
    def _classify_rule_based(self, features: dict):
        hr   = features.get("mean_hr_bpm", 70)
        sdnn = features.get("sdnn_ms", 50)
        if hr > 150:
            label, risk = "Ventricular Tachycardia (suspected)", "HIGH"
        elif hr < 40:
            label, risk = "Bradycardia (suspected)", "HIGH"
        elif hr > 100 and sdnn < 20:
            label, risk = "Atrial Fibrillation (suspected)", "MEDIUM"
        elif sdnn < 20:
            label, risk = "Low HRV — monitoring needed", "MEDIUM"
        elif 60 <= hr <= 100 and sdnn >= 30:
            label, risk = "Normal Sinus Rhythm", "LOW"
        else:
            label, risk = "Inconclusive — further review needed", "MEDIUM"
        return {
            "prediction": label,
            "confidence_pct": 60.0, 
            "risk_level": risk,
            "all_probs": {},
            "model_type": "rule_based_fallback",
        }
# Single shared instance 
classifier = Classifier()
def classify_ecg(features: dict):
    return classifier.classify(features)
