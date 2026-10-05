import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=250):
    """Removes baseline wander (<0.5Hz) and high-freq noise (>40Hz)."""
    nyquist = fs / 2
    low  = lowcut  / nyquist
    high = highcut / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    return filtfilt(b, a, signal)
def notch_filter(signal, freq=50.0, quality_factor=30.0, fs=250):
    """Removes power line interference (50Hz India / 60Hz US)."""
    b, a = iirnotch(w0=freq, Q=quality_factor, fs=fs)
    return filtfilt(b, a, signal)
def normalize(signal):
    """Min-max scale signal to 0-1 range."""
    mn, mx = np.min(signal), np.max(signal)
    if mx - mn == 0:
        return signal
    return (signal - mn) / (mx - mn)

def detect_r_peaks(signal, fs=250):
    """Simplified Pan-Tompkins R-peak detection."""
    diff     = np.diff(signal)
    squared  = diff ** 2
    win_size = max(1, int(0.150 * fs))
    kernel   = np.ones(win_size) / win_size
    smoothed = np.convolve(squared, kernel, mode='same')
    if np.max(smoothed) == 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(
        smoothed,
        distance=int(0.2 * fs),
        height=0.3 * np.max(smoothed)
    )
    return peaks
def segment_beats(signal, r_peaks, fs=250, window_ms=360):
    """Cuts a fixed-length window around each R-peak."""
    half = int((window_ms / 1000) * fs)
    beats = []
    for peak in r_peaks:
        start, end = peak - half, peak + half
        if start < 0 or end > len(signal):
            continue
        beats.append(signal[start:end])
    return beats
def preprocess_ecg(raw_samples, sample_rate=250, apply_notch=True):
    """
    Full preprocessing pipeline — used identically for:
      -  Sample data
      - Live ESP32 streamed data
      - Uploaded CSV files
    """
    signal = np.asarray(raw_samples, dtype=float)

    signal = bandpass_filter(signal, fs=sample_rate)
    if apply_notch:
        signal = notch_filter(signal, fs=sample_rate)
    signal = normalize(signal)

    r_peaks = detect_r_peaks(signal, fs=sample_rate)
    beats   = segment_beats(signal, r_peaks, fs=sample_rate)

    return {
        "clean_signal": signal,
        "r_peaks": r_peaks,
        "beats": beats,
    }
