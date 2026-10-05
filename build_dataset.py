import wfdb
import numpy as np
import pandas as pd
import os

from preprocessing import preprocess_ecg
from feature_extraction import extract_all_features, FEATURE_COLUMNS

# MIT-BIH annotation
MITBIH_LABEL_MAP = {
    'N': "Normal Sinus Rhythm",
    'L': "Left Bundle Branch Block",
    'R': "Right Bundle Branch Block",
    'V': "Premature Ventricular Contraction (PVC)",
    'A': "Premature Atrial Contraction (PAC)",
    'F': "Atrial Flutter",
    '/': "Ventricular Tachycardia",
    'f': "Atrial Fibrillation",
    'j': "ST Elevation (STEMI)",
    'E': "Ventricular Fibrillation",
    'a': "Premature Atrial Contraction (PAC)",
    'J': "First Degree Heart Block",
}
DEFAULT_RECORDS = [
    "100", "101", "103", "105", "106", "108", "109",
    "111", "112", "113", "114", "115", "116", "117",
    "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215",
    "217", "219", "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234",
]
PHYSIONET_DB = "mitdb"  
SAMPLE_RATE  = 360       
def download_record(record_name, local_dir="mit-bih"):
    """Downloads one record (.dat/.hea/.atr) from PhysioNet if not present.
    If the files already exist locally (e.g. you copied them manually,
    or this is a mock/test record), skips downloading entirely."""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, record_name)
    if os.path.exists(local_path + ".dat") and os.path.exists(local_path + ".hea"):
        return local_path   # already present — no need to hit the network
    print(f"  Downloading record {record_name} from PhysioNet...")
    wfdb.dl_database(PHYSIONET_DB, dl_dir=local_dir, records=[record_name])
    return local_path
def load_record(local_path):
    """Loads signal + beat annotations for one record."""
    record     = wfdb.rdrecord(local_path)
    annotation = wfdb.rdann(local_path, 'atr')
    signal  = record.p_signal[:, 0]     # use first lead (usually MLII)
    r_peaks = annotation.sample          # annotated R-peak sample positions
    labels  = annotation.symbol          # beat type symbol at each peak
    return signal, r_peaks, labels
def build_dataset(record_names=DEFAULT_RECORDS, window_sec=10,
                   local_dir="mit-bih", output_csv="ecg_dataset.csv"):
    """
    For each record:
      1. Download (if needed)
      2. Split into `window_sec`-second chunks
      3. Run preprocess_ecg() + extract_all_features() — SAME functions
         used later for live ESP32 data
      4. Label each chunk using majority annotation within that window
      5. Save everything to one CSV
    """
    rows = []
    window_samples = window_sec * SAMPLE_RATE

    for record_name in record_names:
        try:
            local_path = download_record(record_name, local_dir)
            signal, ann_peaks, ann_labels = load_record(local_path)
        except Exception as e:
            print(f"  Skipping {record_name}: {e}")
            continue

        print(f"  Processing record {record_name} "
              f"({len(signal)} samples, {len(ann_peaks)} annotated beats)")

        n_windows = len(signal) // window_samples
        for w in range(n_windows):
            start = w * window_samples
            end   = start + window_samples
            chunk = signal[start:end]
            # Preprocess this chunk (filters + R-peak detect + segment)
            result = preprocess_ecg(chunk, sample_rate=SAMPLE_RATE)
            beats   = result["beats"]
            r_peaks = result["r_peaks"]
            if len(beats) == 0 or len(r_peaks) < 2:
                continue
            features = extract_all_features(beats, r_peaks, fs=SAMPLE_RATE)
            if features is None:
                continue
            # Determine majority label inside this window 
            ann_peaks_arr  = np.asarray(ann_peaks)
            ann_labels_arr = np.asarray(ann_labels)
            mask = (ann_peaks_arr >= start) & (ann_peaks_arr < end)
            window_labels = ann_labels_arr[mask]
            label_counts = {}
            for lbl in window_labels:
                mapped = MITBIH_LABEL_MAP.get(lbl)
                if mapped:
                    label_counts[mapped] = label_counts.get(mapped, 0) + 1
            if not label_counts:
                continue
            majority_label = max(label_counts, key=label_counts.get)
            row = {col: features.get(col, 0.0) for col in FEATURE_COLUMNS}
            row["label"] = majority_label
            row["source_record"] = record_name
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print()
    print(f"Dataset built: {len(df)} rows saved to {output_csv}")
    if len(df) > 0 and "label" in df.columns:
        print("Class distribution:")
        print(df["label"].value_counts())
    else:
        print("WARNING: No rows were generated. Check that record_names is "
              "non-empty and that records downloaded/loaded correctly.")
    return df
if __name__ == "__main__":
    print("================================================")
    print("  CardioSense AI — Building Training Dataset")
    print("  Source: MIT-BIH Arrhythmia Database (PhysioNet)")
    print("================================================")
    print(f"Records to process: {len(DEFAULT_RECORDS)}")
    print("This downloads ~100MB and takes 5-15 minutes on first run.")
    print("------------------------------------------------")
    build_dataset()
