import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

# ============================================================
# Configuration & Constants
# ============================================================

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

    # Non-linear log transform for skewed EDA intensity
    add_feat(features, names, np.log1p(np.maximum(0, row_mean(eda))), "eda_log_mean")

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
# Metric Helpers
# ============================================================

def calc_nmae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    denom = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / (denom + 1e-10)

def calc_nmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    return mse / (var + 1e-10)

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

    print("Extracting domain-engineered features...")
    Z_train, _ = extract_features(X_train_raw, feature_columns)
    del X_train_raw

    # Inner Validation Split (80% Train, 20% Val)
    Z_tr, Z_val, y_tr, y_val = train_test_split(Z_train, y_train, test_size=0.20, random_state=42)

    # Use RobustScaler to handle biosignal artifacts smoothly
    scaler_inner = RobustScaler()
    Z_tr_scaled = scaler_inner.fit_transform(Z_tr)
    Z_val_scaled = scaler_inner.transform(Z_val)

    # Grid search parameters over LinearSVR
    c_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    eps_candidates = [0.0, 0.01]

    print("\n--- Tuning LinearSVR Hyperparameters (C & Epsilon) ---")
    best_nmae = float("inf")
    best_c = 1.0
    best_eps = 0.0

    for c_val in c_candidates:
        for eps_val in eps_candidates:
            model = LinearSVR(
                epsilon=eps_val,
                C=c_val,
                max_iter=4000,
                random_state=42,
                dual="auto"
            )
            model.fit(Z_tr_scaled, y_tr)
            preds = model.predict(Z_val_scaled)
            
            nmae = calc_nmae(y_val, preds)
            nmse = calc_nmse(y_val, preds)
            
            print(f"[C={c_val:<4} | eps={eps_val:<4}] -> Val NMAE: {nmae:.5f} | Val NMSE: {nmse:.5f}")
            
            if nmae < best_nmae:
                best_nmae = nmae
                best_c = c_val
                best_eps = eps_val

    print(f"\nOptimal LinearSVR Found: C={best_c}, epsilon={best_eps} (Best NMAE = {best_nmae:.5f})")

    print("\nScaling full dataset with RobustScaler and retraining optimal LinearSVR...")
    scaler_full = RobustScaler()
    Z_train_scaled = scaler_full.fit_transform(Z_train)
    del Z_train

    final_model = LinearSVR(
        epsilon=best_eps,
        C=best_c,
        max_iter=5000,
        random_state=42,
        dual="auto"
    )
    final_model.fit(Z_train_scaled, y_train)
    del Z_train_scaled, y_train

    print("Loading test data and writing predictions...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    Z_test, _ = extract_features(X_test_raw, feature_columns)
    del X_test_raw

    Z_test_scaled = scaler_full.transform(Z_test)
    del Z_test

    predictions = final_model.predict(Z_test_scaled)
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print(f"Done! Optimized predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()