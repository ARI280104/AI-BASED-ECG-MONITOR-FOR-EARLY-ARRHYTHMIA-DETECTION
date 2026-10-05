#  CardioSense AI — main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from datetime import datetime
import numpy as np
import io
import csv
from preprocessing import preprocess_ecg
from feature_extraction import extract_all_features
from classifier import classify_ecg
app = FastAPI(title="CardioSense AI")
DEFAULT_SAMPLE_RATE = 250   # ESP32 sample rate 
# In-memory store of recent results 
recent_results: List[dict] = []
MAX_HISTORY = 30
def store_result(result: dict):
    recent_results.append(result)
    if len(recent_results) > MAX_HISTORY:
        recent_results.pop(0)
#  CORE ANALYSIS FUNCTION
def analyze_samples(samples: List[float], sample_rate: int,
                     source: str, device_id: str = "unknown"):
    if len(samples) < 50:
        raise HTTPException(status_code=400,
            detail=f"Too few samples ({len(samples)}). Need at least 50.")
    pre = preprocess_ecg(samples, sample_rate=sample_rate)
    signal  = pre["clean_signal"]
    r_peaks = pre["r_peaks"]
    beats   = pre["beats"]
    features = extract_all_features(beats, r_peaks, fs=sample_rate)
    if features is None:
        result = {
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "source": source,
            "device": device_id,
            "samples_received": len(samples),
            "beats_found": int(len(r_peaks)),
            "prediction": "Insufficient beats detected",
            "risk_level": "LOW",
            "confidence_pct": 0,
            "hrv": {},
            "model_type": "n/a",
        }
        store_result(result)
        return result
    clf = classify_ecg(features)
    result = {
        "time": datetime.utcnow().strftime("%H:%M:%S"),
        "source": source,
        "device": device_id,
        "samples_received": len(samples),
        "beats_found": int(len(r_peaks)),
        "prediction": clf["prediction"],
        "risk_level": clf["risk_level"],
        "confidence_pct": clf["confidence_pct"],
        "model_type": clf["model_type"],
        "hrv": {
            "hr_bpm": features.get("mean_hr_bpm"),
            "sdnn_ms": features.get("sdnn_ms"),
            "rmssd_ms": features.get("rmssd_ms"),
            "pnn50_pct": features.get("pnn50_pct"),
        },
        "signal_preview": signal[:300].tolist(),
    }
    store_result(result)
    return result
#CSV UPLOAD
def parse_csv(content: bytes):
    text   = content.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows   = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    start_row = 0
    try:
        float(rows[0][0].strip())
    except (ValueError, IndexError):
        start_row = 1   # first row is a header
    col_index = 1 if len(rows[start_row]) >= 2 else 0
    samples = []
    for row in rows[start_row:]:
        if not row:
            continue
        try:
            samples.append(float(row[col_index].strip()))
        except (ValueError, IndexError):
            continue

    if len(samples) < 50:
        raise HTTPException(status_code=400,
            detail=f"Too few valid numeric samples found ({len(samples)})")
    return samples
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...), sample_rate: int = 250):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files accepted")

    content = await file.read()
    samples = parse_csv(content)

    result = analyze_samples(samples, sample_rate, source="csv_upload",
                              device_id=file.filename)
    return result
#  LIVE ESP32 STREAMING 
class ECGBatch(BaseModel):
    device_id: str
    patient_id: str = "patient-001"
    samples: List[float]
    sample_rate: int = DEFAULT_SAMPLE_RATE
@app.post("/stream_batch")
def stream_batch(payload: ECGBatch):
    result = analyze_samples(
        payload.samples,
        payload.sample_rate,
        source="live_esp32",
        device_id=payload.device_id,
    )
    return result
@app.get("/ping")
def ping():
    return {"status": "online", "time": datetime.utcnow().isoformat()}
@app.get("/latest")
def latest():
    if not recent_results:
        return {"message": "No data yet"}
    return recent_results[-1]
@app.get("/results")
def all_results():
    return {"count": len(recent_results), "results": recent_results}
#  DASHBOARD 
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return open("static/index.html", encoding="utf-8").read()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)