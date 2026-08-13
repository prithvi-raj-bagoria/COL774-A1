import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor


# ============================================================
# Configuration
# ============================================================

HUBER_ALPHA = 1e-4      # Slightly increased L2 penalty to handle correlated spectral features
HUBER_EPSILON = 1.35    # Adjusted robust threshold
HUBER_MAX_ITER = 1000

EXPECTED_RAW_FEATURES = 1640

# ============================================================
# Basic Vectorized Math Helpers
# ============================================================

def row_mean(x):
    return np.mean(x, axis=1, dtype=np.float32)

def row_std(x):
    return np.std(x, axis=1, dtype=np.float32)

def row_rms(x):
    return np.sqrt(np.mean(x * x, axis=1, dtype=np.float32))

def row_slope(x):
    n = x.shape[1]
    t = np.arange(n, dtype=np.float32)
    t = t - np.mean(t)
    denominator = np.sum(t * t)
    return (x @ t) / denominator

def add_feature(features, names, values, name):
    features.append(np.asarray(values, dtype=np.float32))
    names.append(name)

# ============================================================
# Pure NumPy Signal Filtering & High-Precision Spectral Math
# ============================================================

def clean_bvp_signal_fft(bvp, fs=64.0, lowcut=0.7, highcut=3.0):
    """
    Pure NumPy frequency-domain ideal bandpass filter.
    Zeroes out frequencies outside 42 - 180 BPM range and reconstructs clean BVP.
    """
    n_samples = bvp.shape[1]
    bvp_fft = np.fft.rfft(bvp, axis=1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    
    # Zero out non-cardiac frequency bins
    mask = (freqs >= lowcut) & (freqs <= highcut)
    bvp_fft[:, ~mask] = 0.0
    
    # Reconstruct time-domain cleaned signal
    clean_bvp = np.fft.irfft(bvp_fft, n=n_samples, axis=1)
    return clean_bvp.astype(np.float32)

def add_high_precision_spectral_features(features, names, bvp, acc_sq):
    """
    Computes zero-padded FFT for sub-BPM frequency resolution and spectral ratios.
    """
    # 1. BVP Zero-padded FFT for high resolution (10x pad = 0.01 Hz resolution)
    bvp_len = bvp.shape[1]
    pad_len = bvp_len * 10
    
    bvp_fft_complex = np.fft.rfft(bvp, n=pad_len, axis=1)
    bvp_power = np.abs(bvp_fft_complex) ** 2
    bvp_freqs = np.fft.rfftfreq(pad_len, d=1.0 / 64.0)
    
    # Isolate HR Band (0.7 Hz - 3.0 Hz)
    hr_mask = (bvp_freqs >= 0.7) & (bvp_freqs <= 3.0)
    bvp_hr_power = bvp_power[:, hr_mask]
    bvp_hr_freqs = bvp_freqs[hr_mask]
    
    # Peak BPM extraction
    bvp_peak_indices = np.argmax(bvp_hr_power, axis=1)
    bvp_peak_bpm = bvp_hr_freqs[bvp_peak_indices] * 60.0
    add_feature(features, names, bvp_peak_bpm, "fft_bvp_peak_bpm_fine")
    
    # Peak power & Spectral Entropy
    max_bvp_power = bvp_hr_power[np.arange(bvp.shape[0]), bvp_peak_indices]
    add_feature(features, names, max_bvp_power, "fft_bvp_peak_power")
    
    p_bvp = bvp_hr_power / (np.sum(bvp_hr_power, axis=1, keepdims=True) + 1e-10)
    spectral_entropy = -np.sum(p_bvp * np.log2(p_bvp + 1e-10), axis=1)
    add_feature(features, names, spectral_entropy, "fft_bvp_spectral_entropy")
    
    # 2. ACC Zero-padded FFT
    acc_len = acc_sq.shape[1]
    acc_pad_len = acc_len * 10
    
    acc_fft_complex = np.fft.rfft(acc_sq, n=acc_pad_len, axis=1)
    acc_power = np.abs(acc_fft_complex) ** 2
    acc_freqs = np.fft.rfftfreq(acc_pad_len, d=1.0 / 32.0)
    
    acc_mask = (acc_freqs >= 0.7) & (acc_freqs <= 3.0)
    acc_hr_power = acc_power[:, acc_mask]
    acc_hr_freqs = acc_freqs[acc_mask]
    
    acc_peak_indices = np.argmax(acc_hr_power, axis=1)
    acc_peak_bpm = acc_hr_freqs[acc_peak_indices] * 60.0
    add_feature(features, names, acc_peak_bpm, "fft_acc_peak_bpm_fine")
    
    # 3. Motion Artifact Ratio (BVP Power vs ACC Power at the BVP Peak Frequency)
    # Match closest ACC frequency index to BVP peak frequency
    target_freqs = bvp_hr_freqs[bvp_peak_indices]
    acc_target_indices = np.abs(acc_hr_freqs[:, None] - target_freqs[None, :]).argmin(axis=0)
    acc_power_at_bvp_peak = acc_hr_power[np.arange(bvp.shape[0]), acc_target_indices]
    
    # Signal-to-Noise spectral ratio
    spectral_snr = max_bvp_power / (acc_power_at_bvp_peak + 1e-5)
    add_feature(features, names, spectral_snr, "fft_spectral_snr_ratio")
    
    # Absolute frequency delta between Motion and Pulse
    bpm_delta = np.abs(bvp_peak_bpm - acc_peak_bpm)
    add_feature(features, names, bpm_delta, "fft_bpm_delta")
    
    # Motion-Gated HR Estimate
    gated_bpm = bvp_peak_bpm * (1.0 / (1.0 + acc_power_at_bvp_peak * 0.1))
    add_feature(features, names, gated_bpm, "fft_bpm_motion_gated")

# ============================================================
# Time-Domain Cardiac Features (Applied on Cleaned BVP)
# ============================================================

def add_time_domain_cardiac(features, names, clean_bvp):
    vpg = np.diff(clean_bvp, axis=1)
    
    # Zero crossings of derivative -> Peak counting
    crossings = ((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0))
    vpg_bpm = np.sum(crossings, axis=1) * 6.0
    add_feature(features, names, vpg_bpm, "clean_vpg_bpm")
    
    # Autocorrelation on cleaned BVP
    lags = np.asarray([16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76, 86, 96], dtype=np.int32)
    centered = clean_bvp - np.mean(clean_bvp, axis=1, keepdims=True)
    denom = np.sum(centered * centered, axis=1) + 1e-10
    
    autocorrs = []
    for lag in lags:
        num = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        autocorrs.append(num / denom)
        
    autocorrs = np.column_stack(autocorrs)
    best_idx = np.argmax(autocorrs, axis=1)
    best_lag = lags[best_idx]
    
    autocorr_bpm = 3840.0 / best_lag
    add_feature(features, names, autocorr_bpm, "clean_autocorr_bpm")
    add_feature(features, names, autocorrs[np.arange(clean_bvp.shape[0]), best_idx], "clean_autocorr_max_val")

# ============================================================
# Complete Feature Extraction
# ============================================================

def extract_features(X_raw, feature_columns):
    features = []
    names = []

    acc_x_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]
    acc_y_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]
    acc_z_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]
    bvp_idx = [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]
    eda_idx = [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]

    acc_x = X_raw[:, acc_x_idx]
    acc_y = X_raw[:, acc_y_idx]
    acc_z = X_raw[:, acc_z_idx]
    bvp = X_raw[:, bvp_idx]
    eda = X_raw[:, eda_idx]

    # Signal preprocessing
    clean_bvp = clean_bvp_signal_fft(bvp)
    acc_sq = (acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)

    # 1. High-precision spectral domain features
    add_high_precision_spectral_features(features, names, bvp, acc_sq)

    # 2. Cleaned time-domain cardiac features
    add_time_domain_cardiac(features, names, clean_bvp)

    # 3. Accelerometer magnitude / motion metrics
    add_feature(features, names, row_mean(acc_sq), "acc_sq_mean")
    add_feature(features, names, row_std(acc_sq), "acc_sq_std")
    add_feature(features, names, row_rms(acc_sq), "acc_sq_rms")
    add_feature(features, names, row_slope(acc_sq), "acc_sq_slope")
    
    jerk = np.diff(acc_sq, axis=1)
    add_feature(features, names, row_rms(jerk), "acc_jerk_rms")

    # 4. EDA metrics & derivatives
    add_feature(features, names, row_mean(eda), "eda_mean")
    add_feature(features, names, row_std(eda), "eda_std")
    
    eda_vel = np.diff(eda, axis=1)
    add_feature(features, names, row_mean(eda_vel), "eda_vel_mean")
    add_feature(features, names, row_std(eda_vel), "eda_vel_std")

    # 5. Domain interactions
    bvp_peak_bpm_val = features[0]  # "fft_bvp_peak_bpm_fine"
    acc_rms_val = row_rms(acc_sq)
    eda_mean_val = row_mean(eda)

    add_feature(features, names, bvp_peak_bpm_val * acc_rms_val, "interaction_bpm_motion")
    add_feature(features, names, bvp_peak_bpm_val * eda_mean_val, "interaction_bpm_eda")

    # Final matrix compilation
    Z = np.column_stack(features).astype(np.float32)

    if not np.all(np.isfinite(Z)):
        raise ValueError("Feature matrix contains NaN or Inf.")

    return Z, names

# ============================================================
# Main Execution
# ============================================================

def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python3 part_c.py train.csv test.csv predictions.txt")

    train_path, test_path, predictions_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print("Loading training CSV...")
    train_df = pd.read_csv(train_path)

    if "hr" not in train_df.columns:
        raise ValueError("Target column 'hr' not found in training data.")

    feature_columns = [c for c in train_df.columns if c != "hr"]

    if len(feature_columns) != EXPECTED_RAW_FEATURES:
        raise ValueError(f"Expected {EXPECTED_RAW_FEATURES} raw features, got {len(feature_columns)}")

    y_train = train_df["hr"].to_numpy(dtype=np.float64)
    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    del train_df

    print("Creating domain-engineered features...")
    Z_train, feature_names = extract_features(X_train_raw, feature_columns)
    del X_train_raw

    print(f"Feature matrix built successfully: {Z_train.shape}")
    print(f"Total engineered features: {len(feature_names)}")

    print("Scaling features...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    del Z_train

    print("\nFitting HuberRegressor...")
    model = HuberRegressor(
        alpha=HUBER_ALPHA,
        epsilon=HUBER_EPSILON,
        max_iter=HUBER_MAX_ITER,
        tol=1e-3
    )
    model.fit(Z_train_scaled, y_train)

    print(f"Model fitted in {model.n_iter_} iterations.")
    del Z_train_scaled
    del y_train

    print("\nLoading test CSV...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    print("Creating test features...")
    Z_test, _ = extract_features(X_test_raw, feature_columns)
    del X_test_raw

    Z_test_scaled = scaler.transform(Z_test)
    del Z_test

    print("Generating predictions...")
    predictions = model.predict(Z_test_scaled)

    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print(f"Successfully saved {len(predictions)} predictions to {predictions_path}")

if __name__ == "__main__":
    main()