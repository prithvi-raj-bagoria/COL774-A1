import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

# ============================================================
# Configuration & Constants
# ============================================================

HUBER_ALPHA = 1e-4
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 2000
EXPECTED_RAW_FEATURES = 1640

# ============================================================
# Fast Vectorized Signal Helpers (No Python Loops)
# ============================================================

def row_mean(x): return np.mean(x, axis=1, dtype=np.float32)
def row_std(x): return np.std(x, axis=1, dtype=np.float32)
def row_min(x): return np.min(x, axis=1)
def row_max(x): return np.max(x, axis=1)
def row_range(x): return np.ptp(x, axis=1)
def row_rms(x): return np.sqrt(np.mean(x * x, axis=1, dtype=np.float32))

def row_slope(x):
    """Linear trend slope across time samples."""
    n = x.shape[1]
    t = np.arange(n, dtype=np.float32)
    t = t - np.mean(t)
    return (x @ t) / np.sum(t * t)

def row_skewness(x):
    """3rd standardized moment (asymmetry)."""
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - m) / s) ** 3, axis=1, dtype=np.float32)

def row_kurtosis(x):
    """4th standardized moment (tailedness)."""
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - m) / s) ** 4, axis=1, dtype=np.float32)

def row_autocorr_lag(x, lag):
    """Autocorrelation at a specific sample delay (lag)."""
    if x.shape[1] <= lag:
        return np.zeros(x.shape[0], dtype=np.float32)
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    c = x - m
    num = np.sum(c[:, :-lag] * c[:, lag:], axis=1)
    den = np.sum(c * c, axis=1) + 1e-10
    return (num / den).astype(np.float32)

def row_zero_crossings(x):
    """Number of times the centered signal crosses zero."""
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum((c[:, :-1] * c[:, 1:]) < 0, axis=1).astype(np.float32)

def row_local_extrema(x):
    """Count of peak/trough directional changes."""
    d = np.diff(x, axis=1)
    return np.sum((d[:, :-1] * d[:, 1:]) < 0, axis=1).astype(np.float32)

# ============================================================
# Feature Engineering Assembly
# ============================================================

def add_feat(features, names, val, name):
    features.append(np.asarray(val, dtype=np.float32))
    names.append(name)

def add_basic_features(features, names, x, prefix):
    add_feat(features, names, row_mean(x), f"{prefix}_mean")
    add_feat(features, names, row_std(x), f"{prefix}_std")
    add_feat(features, names, row_min(x), f"{prefix}_min")
    add_feat(features, names, row_max(x), f"{prefix}_max")
    add_feat(features, names, row_range(x), f"{prefix}_range")
    add_feat(features, names, row_rms(x), f"{prefix}_rms")
    add_feat(features, names, row_slope(x), f"{prefix}_slope")

def add_deep_features(features, names, x, prefix, autocorr_lags=None):
    add_basic_features(features, names, x, prefix)
    
    # Percentiles
    x_s = np.sort(x, axis=1)
    n = x.shape[1]
    add_feat(features, names, x_s[:, int(0.10 * (n - 1))], f"{prefix}_p10")
    add_feat(features, names, x_s[:, int(0.25 * (n - 1))], f"{prefix}_p25")
    add_feat(features, names, x_s[:, int(0.75 * (n - 1))], f"{prefix}_p75")
    add_feat(features, names, x_s[:, int(0.90 * (n - 1))], f"{prefix}_p90")

    # Higher moments & shape descriptors
    add_feat(features, names, row_skewness(x), f"{prefix}_skew")
    add_feat(features, names, row_kurtosis(x), f"{prefix}_kurt")

    if autocorr_lags:
        for lag in autocorr_lags:
            add_feat(features, names, row_autocorr_lag(x, lag), f"{prefix}_autocorr_{lag}")

    add_feat(features, names, row_zero_crossings(x), f"{prefix}_zero_cross")
    add_feat(features, names, row_local_extrema(x), f"{prefix}_extrema")

def add_temporal_blocks(features, names, x, prefix, n_blocks=4):
    """Splits signal into time sub-windows to capture temporal evolution."""
    bs = x.shape[1] // n_blocks
    for k in range(n_blocks):
        blk = x[:, k * bs:(k + 1) * bs]
        add_feat(features, names, row_mean(blk), f"{prefix}_t{k+1}_mean")
        add_feat(features, names, row_std(blk), f"{prefix}_t{k+1}_std")
        add_feat(features, names, row_slope(blk), f"{prefix}_t{k+1}_slope")

# ============================================================
# Complete Feature Extractor
# ============================================================

def extract_features(X_raw, feature_columns):
    features, names = [], []

    # Slice raw input matrix into channels
    acc_x = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]]
    acc_y = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]]
    acc_z = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]]
    bvp   = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]]
    eda   = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]]

    # 1. BVP Features (Cardiac Signal)
    cardiac_lags = [16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76, 86, 96]
    add_deep_features(features, names, bvp, "bvp", cardiac_lags)
    add_temporal_blocks(features, names, bvp, "bvp")

    # Velocity Photoplethysmogram (VPG) & Estimated BPM
    vpg = np.diff(bvp, axis=1)
    apg = np.diff(vpg, axis=1)
    add_basic_features(features, names, vpg, "vpg")
    add_basic_features(features, names, apg, "apg")

    vpg_crossings = ((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0))
    estimated_bpm = (np.sum(vpg_crossings, axis=1) * 6.0).astype(np.float32)
    add_feat(features, names, estimated_bpm, "vpg_estimated_bpm")

    # 2. Accelerometer Features (Motion Artifacts)
    add_basic_features(features, names, acc_x, "acc_x")
    add_basic_features(features, names, acc_y, "acc_y")
    add_basic_features(features, names, acc_z, "acc_z")

    acc_sq = (acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    add_deep_features(features, names, acc_sq, "acc_sq", [1, 2, 4, 8, 16])
    add_temporal_blocks(features, names, acc_sq, "acc_sq")

    jerk = np.diff(acc_sq, axis=1)
    add_basic_features(features, names, jerk, "jerk")

    # 3. EDA Features (Electrodermal Activity)
    add_deep_features(features, names, eda, "eda", [1, 2, 4])
    add_temporal_blocks(features, names, eda, "eda")

    # 4. Cross-Modal Interactions
    acc_rms_val = row_rms(acc_sq)
    eda_mean_val = row_mean(eda)

    add_feat(features, names, estimated_bpm * acc_rms_val, "inter_bpm_motion")
    add_feat(features, names, row_std(bvp) * row_std(acc_sq), "inter_bvp_motion_var")
    add_feat(features, names, estimated_bpm * eda_mean_val, "inter_bpm_eda")
    add_feat(features, names, acc_rms_val * eda_mean_val, "inter_motion_eda")

    # Signal-to-Noise Proxy
    snr = np.var(bvp, axis=1) / (np.var(acc_sq, axis=1) + 1e-5)
    add_feat(features, names, snr, "time_domain_snr")

    # Stack into 2D array
    Z = np.column_stack(features).astype(np.float32)
    if not np.all(np.isfinite(Z)):
        raise ValueError("Matrix contains NaN or Inf.")
    return Z, names

# ============================================================
# Main Script Execution
# ============================================================

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")

    train_path, test_path, predictions_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    feature_columns = [c for c in train_df.columns if c != "hr"]

    if len(feature_columns) != EXPECTED_RAW_FEATURES:
        raise ValueError(f"Expected {EXPECTED_RAW_FEATURES} raw features, got {len(feature_columns)}")

    y_train = train_df["hr"].to_numpy(dtype=np.float64)
    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    del train_df

    print("Extracting features...")
    Z_train, _ = extract_features(X_train_raw, feature_columns)
    del X_train_raw

    print("Scaling features...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    del Z_train

    print("Training HuberRegressor...")
    model = HuberRegressor(
        alpha=HUBER_ALPHA,
        epsilon=HUBER_EPSILON,
        max_iter=HUBER_MAX_ITER,
        tol=1e-3
    )
    model.fit(Z_train_scaled, y_train)
    del Z_train_scaled, y_train

    print("Loading test data and predicting...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    Z_test, _ = extract_features(X_test_raw, feature_columns)
    del X_test_raw

    Z_test_scaled = scaler.transform(Z_test)
    del Z_test

    predictions = model.predict(Z_test_scaled)
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print(f"Done! Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()