import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

# ============================================================
# Configuration
# ============================================================

HUBER_ALPHA = 1e-4       
HUBER_EPSILON = 1.05     # Optimized for NMAE (Absolute Error)
HUBER_MAX_ITER = 2000    

EXPECTED_RAW_FEATURES = 1640 

# ============================================================
# Basic feature helpers
# ============================================================

def row_mean(x):
    return np.mean(x, axis=1, dtype=np.float32)

def row_std(x):
    return np.std(x, axis=1, dtype=np.float32)

def row_min(x):
    return np.min(x, axis=1)

def row_max(x):
    return np.max(x, axis=1)

def row_range(x):
    return np.ptp(x, axis=1)

def row_rms(x):
    return np.sqrt(np.mean(x * x, axis=1, dtype=np.float32))

def row_slope(x):
    n = x.shape[1]
    t = np.arange(n, dtype=np.float32)
    t = t - np.mean(t)
    denominator = np.sum(t * t)
    return (x @ t) / denominator

def row_skewness(x):
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - mean) / std
    return np.mean(z ** 3, axis=1, dtype=np.float32)

def row_kurtosis(x):
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - mean) / std
    return np.mean(z ** 4, axis=1, dtype=np.float32)

def row_autocorr_lag(x, lag):
    if x.shape[1] <= lag:
        return np.zeros(x.shape[0], dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    centered = x - mean
    numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
    denominator = np.sum(centered * centered, axis=1) + 1e-10
    return (numerator / denominator).astype(np.float32)

def row_zero_crossings(x):
    centered = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum((centered[:, :-1] * centered[:, 1:]) < 0, axis=1).astype(np.float32)

def row_local_extrema(x):
    diff = np.diff(x, axis=1)
    return np.sum((diff[:, :-1] * diff[:, 1:]) < 0, axis=1).astype(np.float32)

# ============================================================
# High-Speed Vectorized Cardiac Autocorrelation (No Python Loops)
# ============================================================

def vectorized_multi_peak_autocorr(signal, fs=64.0, prefix="bvp"):
    """
    Computes sub-sample parabolic interpolated primary & secondary 
    autocorrelation peaks for an entire 2D matrix instantly.
    """
    n_samples = signal.shape[0]
    lags = np.arange(15, 100) # Lags covering ~38 BPM to 256 BPM
    
    mean = np.mean(signal, axis=1, keepdims=True, dtype=np.float32)
    centered = signal - mean
    denom = np.sum(centered * centered, axis=1, keepdims=True) + 1e-10
    
    # 1. Compute 2D Autocorrelations across all lags simultaneously
    R_list = []
    for lag in lags:
        num = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        R_list.append(num)
    R = np.column_stack(R_list) / denom # Shape: (N, 85)
    
    # 2. Vectorized Primary Peak & Sub-sample Parabolic Interpolation
    best_idx = np.argmax(R, axis=1)
    rows = np.arange(n_samples)
    idx_clamped = np.clip(best_idx, 1, len(lags) - 2)
    
    Rm1 = R[rows, idx_clamped - 1]
    R0  = R[rows, idx_clamped]
    Rp1 = R[rows, idx_clamped + 1]
    
    denom_p = 2.0 * (Rm1 - 2.0 * R0 + Rp1)
    denom_safe = np.where(np.abs(denom_p) < 1e-7, 1.0, denom_p)
    delta = np.where(np.abs(denom_p) < 1e-7, 0.0, (Rm1 - Rp1) / denom_safe)
    
    tau_true_1 = lags[idx_clamped] + delta
    bpm_1 = (fs * 60.0) / np.maximum(tau_true_1, 1.0)
    score_1 = R[rows, best_idx]
    
    # 3. Vectorized Secondary Peak Extraction (Harmonic check)
    b_idx = best_idx[:, None]
    lag_indices = np.arange(len(lags))[None, :]
    mask = np.abs(lag_indices - b_idx) <= 5
    R_masked = np.where(mask, -1.0, R)
    
    second_idx = np.argmax(R_masked, axis=1)
    idx_clamped_2 = np.clip(second_idx, 1, len(lags) - 2)
    
    Rm1_2 = R[rows, idx_clamped_2 - 1]
    R0_2  = R[rows, idx_clamped_2]
    Rp1_2 = R[rows, idx_clamped_2 + 1]
    
    denom_p2 = 2.0 * (Rm1_2 - 2.0 * R0_2 + Rp1_2)
    denom_safe2 = np.where(np.abs(denom_p2) < 1e-7, 1.0, denom_p2)
    delta2 = np.where(np.abs(denom_p2) < 1e-7, 0.0, (Rm1_2 - Rp1_2) / denom_safe2)
    
    tau_true_2 = lags[idx_clamped_2] + delta2
    bpm_2 = (fs * 60.0) / np.maximum(tau_true_2, 1.0)
    score_2 = R[rows, second_idx]
    
    return {
        f"{prefix}_bpm_1": bpm_1.astype(np.float32),
        f"{prefix}_score_1": score_1.astype(np.float32),
        f"{prefix}_bpm_2": bpm_2.astype(np.float32),
        f"{prefix}_score_2": score_2.astype(np.float32),
        f"{prefix}_bpm_ratio": (bpm_1 / (bpm_2 + 1e-5)).astype(np.float32)
    }

# ============================================================
# Advanced Time-Domain Physics & Signal Cleanliness
# ============================================================

def compute_tkeo(x):
    """Teager-Kaiser Energy Operator: x[n]^2 - x[n-1]*x[n+1]"""
    x_sq = x[:, 1:-1] ** 2
    x_adj = x[:, :-2] * x[:, 2:]
    return x_sq - x_adj

def add_hjorth_parameters(features, names, x, prefix):
    """Hjorth Activity, Mobility, & Complexity as time-domain frequency proxies."""
    var_x = np.var(x, axis=1) + 1e-10
    dx = np.diff(x, axis=1)
    var_dx = np.var(dx, axis=1) + 1e-10
    ddx = np.diff(dx, axis=1)
    var_ddx = np.var(ddx, axis=1) + 1e-10
    
    mobility_x = np.sqrt(var_dx / var_x)
    mobility_dx = np.sqrt(var_ddx / var_dx)
    complexity = mobility_dx / (mobility_x + 1e-10)
    
    add_feature(features, names, var_x, f"{prefix}_hjorth_activity")
    add_feature(features, names, mobility_x, f"{prefix}_hjorth_mobility")
    add_feature(features, names, complexity, f"{prefix}_hjorth_complexity")

# ============================================================
# Feature assembly helpers
# ============================================================

def add_feature(features, names, values, name):
    features.append(np.asarray(values, dtype=np.float32))
    names.append(name)

def add_percentiles(features, names, x, prefix):
    x_sorted = np.sort(x, axis=1)
    n = x.shape[1]
    
    idx_10 = int(0.10 * (n - 1))
    idx_25 = int(0.25 * (n - 1))
    idx_75 = int(0.75 * (n - 1))
    idx_90 = int(0.90 * (n - 1))

    add_feature(features, names, x_sorted[:, idx_10], f"{prefix}_p10")
    add_feature(features, names, x_sorted[:, idx_25], f"{prefix}_p25")
    add_feature(features, names, x_sorted[:, idx_75], f"{prefix}_p75")
    add_feature(features, names, x_sorted[:, idx_90], f"{prefix}_p90")

def add_basic_features(features, names, x, prefix):
    add_feature(features, names, row_mean(x), f"{prefix}_mean")
    add_feature(features, names, row_std(x), f"{prefix}_std")
    add_feature(features, names, row_min(x), f"{prefix}_min")
    add_feature(features, names, row_max(x), f"{prefix}_max")
    add_feature(features, names, row_range(x), f"{prefix}_range")
    add_feature(features, names, row_rms(x), f"{prefix}_rms")
    add_feature(features, names, row_slope(x), f"{prefix}_slope")

def add_deep_features(features, names, x, prefix, autocorr_lags=None):
    add_basic_features(features, names, x, prefix)
    add_percentiles(features, names, x, prefix)
    add_feature(features, names, row_skewness(x), f"{prefix}_skew")
    add_feature(features, names, row_kurtosis(x), f"{prefix}_kurt")

    if autocorr_lags is not None:
        for lag in autocorr_lags:
            add_feature(features, names, row_autocorr_lag(x, lag), f"{prefix}_autocorr_{lag}")

    add_feature(features, names, row_zero_crossings(x), f"{prefix}_zero_crossings")
    add_feature(features, names, row_local_extrema(x), f"{prefix}_local_extrema")

def add_block_summary(features, names, x, block_size, prefix):
    n_samples = x.shape[1]
    if n_samples % block_size != 0:
        raise ValueError(f"{prefix}: signal length {n_samples} not divisible by {block_size}")

    n_blocks = n_samples // block_size
    blocks = x.reshape(x.shape[0], n_blocks, block_size)
    means = np.mean(blocks, axis=2, dtype=np.float32)
    stds = np.std(blocks, axis=2, dtype=np.float32)

    add_feature(features, names, np.mean(means, axis=1), f"{prefix}_blockmean_mean")
    add_feature(features, names, np.std(means, axis=1), f"{prefix}_blockmean_std")
    add_feature(features, names, row_slope(means), f"{prefix}_blockmean_slope")
    add_feature(features, names, np.mean(stds, axis=1), f"{prefix}_blockstd_mean")

def add_temporal_position_features(features, names, x, prefix, n_blocks=4):
    n_samples = x.shape[1]
    if n_samples % n_blocks != 0:
        raise ValueError(f"{prefix}: cannot split {n_samples} samples into {n_blocks} equal blocks.")

    block_size = n_samples // n_blocks
    for k in range(n_blocks):
        start = k * block_size
        end = start + block_size
        block = x[:, start:end]
        suffix = f"{prefix}_t{k + 1}"
        
        add_feature(features, names, row_mean(block), f"{suffix}_mean")
        add_feature(features, names, row_std(block), f"{suffix}_std")
        add_feature(features, names, row_slope(block), f"{suffix}_slope")


# ============================================================
# Complete feature extraction
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

    # Motion Power & Exponential Motion Mask
    acc_sq = (acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)
    acc_var = np.var(acc_sq, axis=1, dtype=np.float32)
    motion_cleanliness = np.exp(-0.5 * acc_var).astype(np.float32)

    # 1. BVP & Derivatives
    cardiac_lags = [16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76, 86, 96]
    add_deep_features(features, names, bvp, "bvp", cardiac_lags)
    add_block_summary(features, names, bvp, 64, "bvp")
    add_temporal_position_features(features, names, bvp, "bvp")
    add_hjorth_parameters(features, names, bvp, "bvp")

    vpg = np.diff(bvp, axis=1) # 1st Derivative (Velocity)
    apg = np.diff(vpg, axis=1) # 2nd Derivative (Acceleration)
    tkeo_bvp = compute_tkeo(bvp) # TKEO energy

    # Extract Multi-Harmonic Candidate BPMs across multiple signals
    bvp_autocorr = vectorized_multi_peak_autocorr(bvp, fs=64.0, prefix="bvp")
    vpg_autocorr = vectorized_multi_peak_autocorr(vpg, fs=64.0, prefix="vpg")
    tkeo_autocorr = vectorized_multi_peak_autocorr(tkeo_bvp, fs=64.0, prefix="tkeo")

    # Add BPM candidates, confidence scores, and Motion-Gated Interactions
    for ac_dict in [bvp_autocorr, vpg_autocorr, tkeo_autocorr]:
        for k, v in ac_dict.items():
            add_feature(features, names, v, k)
            if "bpm_1" in k:
                # Non-linear basis terms for candidate BPMs
                add_feature(features, names, v ** 2, f"{k}_sq")
                add_feature(features, names, np.sqrt(np.maximum(v, 0.0)), f"{k}_sqrt")
                add_feature(features, names, 1.0 / (v + 1.0), f"{k}_recip")
                # Motion-Gated Candidate: Only trust candidate BPM when wrist is still
                add_feature(features, names, v * motion_cleanliness, f"{k}_gated_motion")

    # 2. Accelerometer Kinematics
    add_basic_features(features, names, acc_x, "acc_x")
    add_basic_features(features, names, acc_y, "acc_y")
    add_basic_features(features, names, acc_z, "acc_z")

    acc_sq_mean = np.mean(acc_sq, axis=1, keepdims=True)
    acc_sq_norm = acc_sq / np.where(acc_sq_mean < 1e-7, 1.0, acc_sq_mean)

    add_deep_features(features, names, acc_sq, "acc_sq", [1, 2, 4, 8, 16])
    add_basic_features(features, names, acc_sq_norm, "acc_sq_norm")
    add_block_summary(features, names, acc_sq, 32, "acc_sq")
    add_hjorth_parameters(features, names, acc_sq, "acc_sq")

    jerk = np.diff(acc_sq, axis=1)
    add_basic_features(features, names, jerk, "jerk")

    # 3. EDA
    add_deep_features(features, names, eda, "eda", [1, 2, 4])
    add_block_summary(features, names, eda, 4, "eda")
    add_temporal_position_features(features, names, eda, "eda")

    # 4. Interactions & Quality Indices
    bvp_var = np.var(bvp, axis=1, dtype=np.float32)
    snr_time = bvp_var / (acc_var + 1e-5)
    add_feature(features, names, snr_time, "snr_time_domain")
    add_feature(features, names, motion_cleanliness, "motion_cleanliness_mask")

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
        raise SystemExit(
            "Usage:\n"
            "python3 part_c.py train.csv test.csv predictions.txt"
        )

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    predictions_path = sys.argv[3]

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

    print("Extracting high-speed non-linear physiological features...")
    Z_train, feature_names = extract_features(X_train_raw, feature_columns)
    del X_train_raw

    print(f"Feature matrix built successfully: {Z_train.shape}")

    print("Fitting StandardScaler on training data...")
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

    print(f"Huber model fitted in {model.n_iter_} iterations.")

    del Z_train_scaled
    del y_train

    print("\nLoading test CSV...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    print("Creating test engineered features...")
    Z_test, test_feature_names = extract_features(X_test_raw, feature_columns)
    del X_test_raw

    Z_test_scaled = scaler.transform(Z_test)
    del Z_test

    print("Generating predictions...")
    predictions = model.predict(Z_test_scaled)

    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print(f"Successfully saved {len(predictions)} predictions to {predictions_path}")

if __name__ == "__main__":
    main()