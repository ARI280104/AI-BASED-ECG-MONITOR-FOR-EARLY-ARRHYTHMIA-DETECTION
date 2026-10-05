import numpy as np
def extract_hrv_features(r_peaks, fs=250):
    """Heart Rate Variability features from R-peak positions."""
    if len(r_peaks) < 2:
        return {}
    rr_ms = np.diff(r_peaks) / fs * 1000   # RR intervals in ms
    mean_hr = 60000 / np.mean(rr_ms)
    min_hr  = 60000 / np.max(rr_ms)
    max_hr  = 60000 / np.min(rr_ms)
    sdnn = np.std(rr_ms)
    successive_diffs = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else 0.0
    if len(successive_diffs) > 0:
        nn50  = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = (nn50 / len(successive_diffs)) * 100
    else:
        pnn50 = 0.0
    return {
        "mean_rr_ms":  round(float(np.mean(rr_ms)), 2),
        "mean_hr_bpm": round(float(mean_hr), 2),
        "min_hr_bpm":  round(float(min_hr), 2),
        "max_hr_bpm":  round(float(max_hr), 2),
        "sdnn_ms":     round(float(sdnn), 2),
        "rmssd_ms":    round(float(rmssd), 2),
        "pnn50_pct":   round(float(pnn50), 2),
    }
def extract_morphological_features(beat, fs=250):
    """P/Q/R/S/T wave features from a single beat window centered on R-peak."""
    n = len(beat)
    center = n // 2
    ms_per_sample = 1000 / fs
    r_amp = beat[center]
    q_start = max(0, center - int(0.05 * fs))
    q_idx   = q_start + int(np.argmin(beat[q_start:center])) if center > q_start else center
    q_amp   = beat[q_idx]
    s_end   = min(n, center + int(0.05 * fs))
    s_idx   = center + int(np.argmin(beat[center:s_end])) if s_end > center else center
    s_amp   = beat[s_idx]
    qrs_duration_ms = (s_idx - q_idx) * ms_per_sample
    p_start = max(0, center - int(0.25 * fs))
    p_end   = max(p_start + 1, center - int(0.10 * fs))
    p_idx   = p_start + int(np.argmax(beat[p_start:p_end]))
    p_amp   = beat[p_idx]
    t_start = min(n - 1, center + int(0.15 * fs))
    t_end   = min(n, center + int(0.40 * fs))
    if t_end > t_start:
        t_idx = t_start + int(np.argmax(beat[t_start:t_end]))
    else:
        t_idx = t_start
    t_amp = beat[t_idx]
    pr_interval_ms = (center - p_idx) * ms_per_sample
    qt_interval_ms = (t_idx - q_idx) * ms_per_sample
    st_idx   = min(n - 1, center + int(0.08 * fs))
    st_level = beat[st_idx]
    return {
        "r_amplitude":     round(float(r_amp), 4),
        "q_amplitude":     round(float(q_amp), 4),
        "s_amplitude":     round(float(s_amp), 4),
        "p_amplitude":     round(float(p_amp), 4),
        "t_amplitude":     round(float(t_amp), 4),
        "qrs_duration_ms": round(float(qrs_duration_ms), 2),
        "pr_interval_ms":  round(float(pr_interval_ms), 2),
        "qt_interval_ms":  round(float(qt_interval_ms), 2),
        "st_level":        round(float(st_level), 4),
    }
def extract_all_features(beats, r_peaks, fs=250):
    """
    Combines HRV (from full window) + averaged morphological
    features (across all beats) into ONE flat feature vector.
    This exact same function is used for training AND live prediction.
    """
    hrv = extract_hrv_features(r_peaks, fs)
    if not hrv:
        # Not enough peaks to compute HRV — return empty so caller can skip
        return None
    if len(beats) == 0:
        return None
    all_morph = [extract_morphological_features(b, fs) for b in beats]
    avg_morph = {}
    for key in all_morph[0].keys():
        values = [beat[key] for beat in all_morph]
        avg_morph[f"avg_{key}"] = round(float(np.mean(values)), 4)
    return {**hrv, **avg_morph}
# Fixed column order
FEATURE_COLUMNS = [
    "mean_rr_ms", "mean_hr_bpm", "min_hr_bpm", "max_hr_bpm",
    "sdnn_ms", "rmssd_ms", "pnn50_pct",
    "avg_r_amplitude", "avg_q_amplitude", "avg_s_amplitude",
    "avg_p_amplitude", "avg_t_amplitude",
    "avg_qrs_duration_ms", "avg_pr_interval_ms", "avg_qt_interval_ms",
    "avg_st_level",
]
def features_to_vector(features: dict):
    """Converts a feature dict into a fixed-order numpy array for the model."""
    return np.array([features.get(col, 0.0) for col in FEATURE_COLUMNS])
