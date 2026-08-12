import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

# ============================================================
# Configuration
# ============================================================

# Search space for RidgeCV hyperparameter tuning
ALPHAS = np.logspace(-3, 5, 60)

# ============================================================
# Vectorized Feature Extraction Helpers
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

def row_skewness(x):
    """3rd statistical moment: measures signal asymmetry."""
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - mean) / std
    return np.mean(z**3, axis=1, dtype=np.float32)

def row_kurtosis(x):
    """4th statistical moment: measures peak sharpness."""
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - mean) / std
    return np.mean(z**4, axis=1, dtype=np.float32)

def row_slope(x):
    """Least-squares linear trend/slope against time."""
    n = x.shape[1]
    t = np.arange(n, dtype=np.float32)
    t = t - np.mean(t)
    denominator = np.sum(t * t)
    return (x @ t) / denominator

def row_autocorr_lag(x, lag):
    """Normalized autocorrelation at lag k."""
    if x.shape[1] <= lag:
        return np.zeros(x.shape[0], dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    centered = x - mean
    numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
    denominator = np.sum(centered * centered, axis=1) + 1e-10
    return (numerator / denominator).astype(np.float32)

def row_zero_crossings(x):
    """Counts mean-centered zero crossings (frequency proxy)."""
    centered = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum((centered[:, :-1] * centered[:, 1:]) < 0, axis=1).astype(np.float32)

def row_local_extrema(x):
    """Counts peaks and valleys via first-derivative sign changes."""
    diff = np.diff(x, axis=1)
    return np.sum((diff[:, :-1] * diff[:, 1:]) < 0, axis=1).astype(np.float32)

def row_cross_correlation(x, y):
    """Row-wise Pearson correlation between two aligned signals."""
    x_centered = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    y_centered = y - np.mean(y, axis=1, keepdims=True, dtype=np.float32)
    num = np.sum(x_centered * y_centered, axis=1)
    den = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)) + 1e-10
    return (num / den).astype(np.float32)

def row_vpg_estimated_bpm(bvp):
    """
    Estimates baseline heart rate (BPM) by counting positive-going zero crossings
    in Velocity PPG (VPG) over the 10-second window.
    """
    vpg = np.diff(bvp, axis=1)
    pos_crossings = (vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0)
    beat_count = np.sum(pos_crossings, axis=1)
    return (beat_count * 6.0).astype(np.float32)  # 10s window * 6 = BPM estimate


# ============================================================
# Modular Feature Assembly
# ============================================================

def add_feature(features, names, values, name):
    features.append(values.astype(np.float32))
    names.append(name)

def add_percentiles(features, names, x, prefix):
    """Extracts 10th, 25th, 75th, and 90th percentiles."""
    p10, p25, p75, p90 = np.percentile(x, [10, 25, 75, 90], axis=1)
    add_feature(features, names, p10, f"{prefix}_p10")
    add_feature(features, names, p25, f"{prefix}_p25")
    add_feature(features, names, p75, f"{prefix}_p75")
    add_feature(features, names, p90, f"{prefix}_p90")

def add_basic_signal_features(features, names, x, prefix):
    """Fast statistical summaries for auxiliary signals."""
    add_feature(features, names, row_mean(x), f"{prefix}_mean")
    add_feature(features, names, row_std(x), f"{prefix}_std")
    add_feature(features, names, row_min(x), f"{prefix}_min")
    add_feature(features, names, row_max(x), f"{prefix}_max")
    add_feature(features, names, row_range(x), f"{prefix}_range")
    add_feature(features, names, row_rms(x), f"{prefix}_rms")
    add_feature(features, names, row_slope(x), f"{prefix}_slope")

def add_deep_signal_features(features, names, x, prefix, autocorr_lags=None):
    """Full shape, moment, and temporal features for primary signals."""
    add_basic_signal_features(features, names, x, prefix)
    add_percentiles(features, names, x, prefix)
    add_feature(features, names, row_skewness(x), f"{prefix}_skew")
    add_feature(features, names, row_kurtosis(x), f"{prefix}_kurt")
    
    if autocorr_lags is not None:
        for lag in autocorr_lags:
            add_feature(features, names, row_autocorr_lag(x, lag), f"{prefix}_autocorr_lag{lag}")
            
    add_feature(features, names, row_zero_crossings(x), f"{prefix}_zero_crossings")
    add_feature(features, names, row_local_extrema(x), f"{prefix}_local_extrema")

def block_features(x, block_size, prefix, features, names):
    """Captures intra-window signal variation across sub-blocks."""
    n_samples = x.shape[1]
    if n_samples % block_size != 0:
        return
    n_blocks = n_samples // block_size
    blocks = x.reshape(x.shape[0], n_blocks, block_size)

    means = np.mean(blocks, axis=2, dtype=np.float32)
    stds = np.std(blocks, axis=2, dtype=np.float32)

    add_feature(features, names, np.mean(means, axis=1), f"{prefix}_blockmean_mean")
    add_feature(features, names, np.std(means, axis=1), f"{prefix}_blockmean_std")  # Fixed typo
    add_feature(features, names, row_slope(means), f"{prefix}_blockmean_slope")
    add_feature(features, names, np.mean(stds, axis=1), f"{prefix}_blockstd_mean")


def extract_features_from_array(X_raw, feature_columns):
    """Extracts domain-engineered features using fast vectorized operations."""
    features, names = [], []

    acc_x_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]
    acc_y_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]
    acc_z_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]
    bvp_idx = [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]
    eda_idx = [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]

    # --- 1. BVP & Derivatives ---
    bvp = X_raw[:, bvp_idx]
    
    # Target cardiac beat periods at 64 Hz (lags 16 to 96 sample offsets -> 40 to 180 BPM)
    cardiac_lags = [16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76, 86, 96]
    add_deep_signal_features(features, names, bvp, "bvp", autocorr_lags=cardiac_lags)
    block_features(bvp, 64, "bvp", features, names)

    # Explicit baseline BPM proxy via VPG zero-crossing rate
    add_feature(features, names, row_vpg_estimated_bpm(bvp), "vpg_estimated_bpm")

    vpg = np.diff(bvp, axis=1)  # Velocity PPG
    add_basic_signal_features(features, names, vpg, "vpg")

    apg = np.diff(vpg, axis=1)  # Acceleration PPG
    add_basic_signal_features(features, names, apg, "apg")

    # --- 2. Accelerometer Features ---
    acc_x = X_raw[:, acc_x_idx]
    acc_y = X_raw[:, acc_y_idx]
    acc_z = X_raw[:, acc_z_idx]

    add_basic_signal_features(features, names, acc_x, "acc_x")
    add_basic_signal_features(features, names, acc_y, "acc_y")
    add_basic_signal_features(features, names, acc_z, "acc_z")

    acc_sq = (acc_x**2 + acc_y**2 + acc_z**2)
    acc_sq_mean = np.mean(acc_sq, axis=1, keepdims=True)
    acc_sq_norm = acc_sq / np.where(acc_sq_mean < 1e-7, 1.0, acc_sq_mean)

    acc_lags = [1, 2, 4, 8, 16]
    add_deep_signal_features(features, names, acc_sq, "acc_sq", autocorr_lags=acc_lags)
    add_basic_signal_features(features, names, acc_sq_norm, "acc_sq_norm")
    block_features(acc_sq, 32, "acc_sq", features, names)

    jerk = np.diff(acc_sq, axis=1)
    add_basic_signal_features(features, names, jerk, "jerk")

    # --- 3. Cross-Modality Motion Correlation ---
    bvp_downsampled = bvp[:, ::2]  # Match 32Hz ACC sample size
    

    # --- 4. EDA Features ---
    eda = X_raw[:, eda_idx]
    eda_lags = [1, 2, 4]
    add_deep_signal_features(features, names, eda, "eda", autocorr_lags=eda_lags)
    block_features(eda, 4, "eda", features, names)

    # Stack all extracted features into float32 matrix
    Z = np.column_stack(features).astype(np.float32)

    if not np.all(np.isfinite(Z)):
        raise ValueError("Feature matrix contains NaN or Infinite values.")

    return Z, names


# ============================================================
# Main Execution Pipeline
# ============================================================

def main():
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python3 part_c.py train.csv test.csv predictions.txt")

    train_path, test_path, predictions_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # --- Load Training Data ---
    print("Loading training CSV...")
    train_df = pd.read_csv(train_path)
    if "hr" not in train_df.columns:
        raise ValueError("Target column 'hr' not found in training dataset.")

    feature_columns = [col for col in train_df.columns if col != "hr"]
    y_train = train_df["hr"].to_numpy(dtype=np.float64)

    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    del train_df

    print("Extracting domain-engineered training features...")
    Z_train, feature_names = extract_features_from_array(X_train_raw, feature_columns)
    del X_train_raw

    print(f"Extracted {Z_train.shape[1]} features across {Z_train.shape[0]} training samples.")

    # --- Model Training (Fast LOO Cross-Validation) ---
    print("\nFitting RidgeCV model (Fast Generalized Cross-Validation)...")
    model = make_pipeline(
        StandardScaler(), 
        RidgeCV(
        alphas=ALPHAS,
        scoring="neg_mean_absolute_error"
    )
    )
    model.fit(Z_train, y_train)

    best_alpha = model.named_steps["ridgecv"].alpha_
    print(f"Model trained successfully. Optimal Regularization Alpha: {best_alpha:.4f}")

    del Z_train

    # --- Load Test Data & Predict ---
    print("\nLoading test CSV...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    print("Extracting test set features...")
    Z_test, test_feature_names = extract_features_from_array(X_test_raw, feature_columns)
    del X_test_raw

    if test_feature_names != feature_names:
        raise ValueError("Training and test feature alignments differ.")

    print("Generating predictions...")
    predictions = model.predict(Z_test)

    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print(f"Successfully saved {len(predictions)} predictions to {predictions_path}")

if __name__ == "__main__":
    main()