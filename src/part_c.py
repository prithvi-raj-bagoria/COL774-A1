import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

# ============================================================
# Configuration
# ============================================================

HUBER_ALPHA = 1e-4       
HUBER_EPSILON = 1.05     # Lowered to 1.05 to heavily optimize for Absolute Error (NMAE)
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
# Advanced Time-Domain Features (Replacing FFT)
# ============================================================

def add_tkeo_features(features, names, x, prefix):
    """Teager-Kaiser Energy Operator (TKEO) to isolate systolic peaks."""
    # x[n]^2 - x[n-1]*x[n+1]
    x_sq = x[:, 1:-1] ** 2
    x_adj = x[:, :-2] * x[:, 2:]
    tkeo = x_sq - x_adj
    
    add_feature(features, names, np.mean(tkeo, axis=1), f"{prefix}_tkeo_mean")
    add_feature(features, names, np.std(tkeo, axis=1), f"{prefix}_tkeo_std")
    add_feature(features, names, np.max(tkeo, axis=1), f"{prefix}_tkeo_max")
    add_feature(features, names, row_zero_crossings(tkeo), f"{prefix}_tkeo_zero_cross")

def add_hjorth_parameters(features, names, x, prefix):
    """Hjorth Parameters for time-domain frequency estimation."""
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

def dominant_cardiac_period_interpolated(bvp):
    """Sub-Sample Parabolic Interpolation of Autocorrelation for continuous BPM."""
    # Lags from 15 to 100 cover roughly 38 BPM to 256 BPM at 64Hz
    lags = np.arange(15, 100)
    mean = np.mean(bvp, axis=1, keepdims=True, dtype=np.float32)
    centered = bvp - mean
    denominator = np.sum(centered * centered, axis=1) + 1e-10
    
    autocorrelations = []
    for lag in lags:
        numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        autocorrelations.append(numerator / denominator)
        
    R = np.column_stack(autocorrelations)
    best_idx = np.argmax(R, axis=1)
    
    tau_true = np.zeros(bvp.shape[0], dtype=np.float32)
    peak_ac = np.zeros(bvp.shape[0], dtype=np.float32)
    
    for i in range(bvp.shape[0]):
        idx = best_idx[i]
        peak_ac[i] = R[i, idx]
        
        # Apply Parabolic Interpolation if not at the boundaries
        if 0 < idx < len(lags) - 1:
            Rm1 = R[i, idx - 1]
            R0 = R[i, idx]
            Rp1 = R[i, idx + 1]
            
            denom = 2 * (Rm1 - 2 * R0 + Rp1)
            if denom != 0:
                delta = (Rm1 - Rp1) / denom
            else:
                delta = 0
            tau_true[i] = lags[idx] + delta
        else:
            tau_true[i] = lags[idx]
            
    bpm = (3840.0 / (tau_true + 1e-7)) # 60 * 64Hz = 3840
    return bpm.astype(np.float32), peak_ac.astype(np.float32)

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

# ============================================================
# BVP / cardiac features
# ============================================================

def vpg_bpm(bvp):
    vpg = np.diff(bvp, axis=1)
    crossings = ((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0))
    beat_count = np.sum(crossings, axis=1)
    return (beat_count * 6.0).astype(np.float32)

def add_vpg_features(features, names, bvp):
    vpg = np.diff(bvp, axis=1)
    apg = np.diff(vpg, axis=1)

    add_basic_features(features, names, vpg, "vpg")
    add_basic_features(features, names, apg, "apg")
    add_feature(features, names, vpg_bpm(bvp), "vpg_estimated_bpm")

    # Replaced basic lag estimation with High-Precision Parabolic Interpolation
    dominant_bpm, dominant_ac = dominant_cardiac_period_interpolated(bvp)
    add_feature(features, names, dominant_bpm, "bvp_dominant_bpm_interp")
    add_feature(features, names, dominant_ac, "bvp_dominant_autocorr_interp")

    positive_energy = np.mean(np.maximum(vpg, 0.0) ** 2, axis=1)
    negative_energy = np.mean(np.minimum(vpg, 0.0) ** 2, axis=1) + 1e-10

    add_feature(features, names, positive_energy / negative_energy, "bvp_rise_fall_energy_ratio")
    add_feature(features, names, np.mean(np.maximum(vpg, 0.0), axis=1), "bvp_rise_strength")
    add_feature(features, names, np.mean(np.abs(np.minimum(vpg, 0.0)), axis=1), "bvp_fall_strength")

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

    # ========================================================
    # 1. BVP
    # ========================================================
    cardiac_lags = [16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76, 86, 96]
    add_deep_features(features, names, bvp, "bvp", cardiac_lags)
    add_block_summary(features, names, bvp, 64, "bvp")
    add_vpg_features(features, names, bvp)
    add_temporal_position_features(features, names, bvp, "bvp")
    
    # Advanced BVP Time-Domain Physics
    add_tkeo_features(features, names, bvp, "bvp")
    add_hjorth_parameters(features, names, bvp, "bvp")

    # ========================================================
    # 2. ACCELEROMETER
    # ========================================================
    add_basic_features(features, names, acc_x, "acc_x")
    add_basic_features(features, names, acc_y, "acc_y")
    add_basic_features(features, names, acc_z, "acc_z")

    acc_sq = (acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)
    acc_sq_mean = np.mean(acc_sq, axis=1, keepdims=True)
    acc_sq_norm = acc_sq / np.where(acc_sq_mean < 1e-7, 1.0, acc_sq_mean)

    add_deep_features(features, names, acc_sq, "acc_sq", [1, 2, 4, 8, 16])
    add_basic_features(features, names, acc_sq_norm, "acc_sq_norm")
    add_block_summary(features, names, acc_sq, 32, "acc_sq")
    
    # Advanced ACC Kinematics
    add_hjorth_parameters(features, names, acc_sq, "acc_sq")

    jerk = np.diff(acc_sq, axis=1)
    add_basic_features(features, names, jerk, "jerk")

    acc_sq_sorted = np.sort(acc_sq, axis=1)
    idx_10 = int(0.10 * (acc_sq.shape[1] - 1))
    idx_90 = int(0.90 * (acc_sq.shape[1] - 1))
    burstiness = acc_sq_sorted[:, idx_90] - acc_sq_sorted[:, idx_10]
    add_feature(features, names, burstiness, "acc_sq_burstiness")

    second_diff = np.diff(acc_sq, n=2, axis=1)
    add_feature(features, names, np.mean(np.abs(second_diff), axis=1), "acc_sq_second_diff_abs_mean")
    add_feature(features, names, np.std(second_diff, axis=1), "acc_sq_second_diff_std")
    add_temporal_position_features(features, names, acc_sq, "acc_sq")

    # ========================================================
    # 3. EDA
    # ========================================================
    add_deep_features(features, names, eda, "eda", [1, 2, 4])
    add_block_summary(features, names, eda, 4, "eda")
    add_temporal_position_features(features, names, eda, "eda")

    # ========================================================
    # 4. Physiologically Motivated Interactions
    # ========================================================
    bvp_bpm_val = vpg_bpm(bvp)
    acc_rms_val = row_rms(acc_sq)
    eda_mean_val = row_mean(eda)

    add_feature(features, names, bvp_bpm_val * acc_rms_val, "interaction_bpm_motion")
    add_feature(features, names, row_std(bvp) * row_std(acc_sq), "interaction_bvp_motion_variability")
    add_feature(features, names, bvp_bpm_val * eda_mean_val, "interaction_bpm_eda")
    add_feature(features, names, acc_rms_val * eda_mean_val, "interaction_motion_eda")
    
    # SNR equivalent (Variance Ratios)
    acc_var = np.var(acc_sq, axis=1)
    bvp_var = np.var(bvp, axis=1)
    snr_time_domain = bvp_var / (acc_var + 1e-5)
    add_feature(features, names, snr_time_domain, "time_domain_bvp_acc_snr")

    # --------------------------------------------------------
    # Final matrix compilation
    # --------------------------------------------------------
    Z = np.column_stack(features).astype(np.float32)

    if not np.all(np.isfinite(Z)):
        raise ValueError("Feature matrix contains NaN or Inf.")

    return Z, names


# ============================================================
# Main
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

    print("Creating comprehensive domain-engineered features...")
    Z_train, feature_names = extract_features(X_train_raw, feature_columns)
    del X_train_raw

    print(f"Feature matrix built successfully: {Z_train.shape}")
    print(f"Total number of engineered features: {len(feature_names)}")

    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    del Z_train

    print("\nFitting final HuberRegressor...")
    model = HuberRegressor(
        alpha=HUBER_ALPHA,
        epsilon=HUBER_EPSILON,
        max_iter=HUBER_MAX_ITER,
        tol=1e-3  
    )
    model.fit(Z_train_scaled, y_train)

    print("Huber model fitted successfully.")
    print(f"Iterations used = {model.n_iter_}")

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