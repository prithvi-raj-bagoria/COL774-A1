import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold

# ============================================================
# 1. Configuration & Hyperparameters
# ============================================================
EXPECTED_RAW_FEATURES = 1640
CV_FOLDS = 5
RANDOM_STATE = 42

# ============================================================
# 2. Terminal Output Helpers
# ============================================================
def print_header(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)

def print_step(step_num, total_steps, description):
    print(f"\n▶ [{step_num}/{total_steps}] {description}")

def print_stat(label, value):
    print(f"    ➜ {label:<20} : {value}")

def print_time(start_time):
    print(f"    ⏱  Time taken          : {time.perf_counter() - start_time:.2f}s")

# ============================================================
# 3. Fast Vectorized Signal Math (Row-wise operations)
# ============================================================
def row_mean(x): return np.mean(x, axis=1, dtype=np.float32)
def row_std(x): return np.std(x, axis=1, dtype=np.float32)
def row_min(x): return np.min(x, axis=1)
def row_max(x): return np.max(x, axis=1)
def row_range(x): return np.ptp(x, axis=1)
def row_rms(x): return np.sqrt(np.mean(x * x, axis=1, dtype=np.float32))

def row_slope(x):
    t = np.arange(x.shape[1], dtype=np.float32) - np.mean(np.arange(x.shape[1], dtype=np.float32))
    return (x @ t) / (np.sum(t * t) + 1e-7)

def row_skewness(x):
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - mean) / std) ** 3, axis=1, dtype=np.float32)

def row_kurtosis(x):
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - mean) / std) ** 4, axis=1, dtype=np.float32)

def row_autocorr_lag(x, lag):
    if x.shape[1] <= lag: return np.zeros(x.shape[0], dtype=np.float32)
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return (np.sum(c[:, :-lag] * c[:, lag:], axis=1) / (np.sum(c * c, axis=1) + 1e-10)).astype(np.float32)

def row_zero_crossings(x):
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum((c[:, :-1] * c[:, 1:]) < 0, axis=1).astype(np.float32)

def row_local_extrema(x):
    d = np.diff(x, axis=1)
    return np.sum((d[:, :-1] * d[:, 1:]) < 0, axis=1).astype(np.float32)

def row_sma(x, y, z):
    """Signal Magnitude Area: Actigraphy standard for physical exertion."""
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)

# ============================================================
# 4. Feature Assembly Framework
# ============================================================
def add_feat(features, names, values, name):
    features.append(np.asarray(values, dtype=np.float32))
    names.append(name)

def add_basic_features(features, names, x, prefix):
    for val, suffix in zip(
        [row_mean(x), row_std(x), row_min(x), row_max(x), row_range(x), row_rms(x), row_slope(x)],
        ["mean", "std", "min", "max", "range", "rms", "slope"]
    ): add_feat(features, names, val, f"{prefix}_{suffix}")

def add_deep_features(features, names, x, prefix):
    """Full morphological blueprint (basics + percentiles + shape)."""
    add_basic_features(features, names, x, prefix)
    sorted_x, n = np.sort(x, axis=1), x.shape[1]
    
    for p, suffix in zip([0.10, 0.25, 0.75, 0.90], ["p10", "p25", "p75", "p90"]):
        add_feat(features, names, sorted_x[:, int(p * (n - 1))], f"{prefix}_{suffix}")
        
    add_feat(features, names, row_skewness(x), f"{prefix}_skew")
    add_feat(features, names, row_kurtosis(x), f"{prefix}_kurt")

def add_temporal_blocks(features, names, x, prefix, n_blocks=4):
    """Splits signal into 4 sub-windows to capture changes over time."""
    bs = x.shape[1] // n_blocks
    for k in range(n_blocks):
        block = x[:, k * bs:(k + 1) * bs]
        add_feat(features, names, row_mean(block), f"{prefix}_t{k + 1}_mean")
        add_feat(features, names, row_std(block), f"{prefix}_t{k + 1}_std")
        add_feat(features, names, row_slope(block), f"{prefix}_t{k + 1}_slope")

# ============================================================
# 5. Core Feature Extraction (The Bio-Mathematical Blueprint)
# ============================================================
def extract_features(X_raw, feature_columns):
    features, names = [], []
    
    acc_x = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]]
    acc_y = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]]
    acc_z = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]]
    bvp = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]]
    eda = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]]

    # --- 1. Cardiac Features & The "Frequency Equalizer" ---
    add_deep_features(features, names, bvp, "bvp")
    add_feat(features, names, row_zero_crossings(bvp), "bvp_zero_cross")
    add_feat(features, names, row_local_extrema(bvp), "bvp_extrema")
    add_temporal_blocks(features, names, bvp, "bvp")

    # The Equalizer: Autocorrelation at specific physiological lags (50 BPM to 170 BPM)
    bpm_targets = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
    for bpm in bpm_targets:
        lag = int(60.0 * 64 / bpm)
        add_feat(features, names, row_autocorr_lag(bvp, lag), f"bvp_autocorr_{bpm}bpm")

    # --- 2. Pulse Morphology (VPG & APG Deep Stats) ---
    vpg, apg = np.diff(bvp, axis=1), np.diff(np.diff(bvp, axis=1), axis=1)
    add_deep_features(features, names, vpg, "vpg")
    add_deep_features(features, names, apg, "apg")

    # Fallback BPM estimator from zero-crossings of velocity
    estimated_bpm = (np.sum((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0), axis=1) * 6.0).astype(np.float32)
    add_feat(features, names, estimated_bpm, "vpg_estimated_bpm")

    # --- 3. Motion Context ---
    add_basic_features(features, names, acc_x, "acc_x")
    add_basic_features(features, names, acc_y, "acc_y")
    add_basic_features(features, names, acc_z, "acc_z")
    
    # Actigraphy Energy (SMA & Squared)
    acc_sma_val = row_sma(acc_x, acc_y, acc_z)
    add_feat(features, names, acc_sma_val, "acc_sma")
    
    acc_sq = acc_x ** 2 + acc_y ** 2 + acc_z ** 2
    add_deep_features(features, names, acc_sq, "acc_sq")
    add_temporal_blocks(features, names, acc_sq, "acc_sq")

    # --- 4. Stress/Electrodermal Context ---
    add_deep_features(features, names, eda, "eda")
    add_temporal_blocks(features, names, eda, "eda")
    
    phasic_eda = np.diff(eda, axis=1)
    add_basic_features(features, names, phasic_eda, "phasic_eda")

    # --- 5. The "Artifact Gate" & Cross-Modal Interactions ---
    bvp_range_val = row_range(bvp)
    
    # Gate 1: Strong pulse vs. high motion
    add_feat(features, names, bvp_range_val * acc_sma_val, "inter_bvp_range_motion")
    
    # Gate 2: VPG Estimate vs. Motion Standard Deviation
    acc_sq_std = row_std(acc_sq)
    add_feat(features, names, estimated_bpm * acc_sq_std, "inter_bpm_motion_var")
    
    # Gate 3: Motion vs. Stress
    eda_mean_val = row_mean(eda)
    add_feat(features, names, acc_sma_val * eda_mean_val, "inter_motion_eda")

    Z = np.column_stack(features).astype(np.float32)
    if not np.all(np.isfinite(Z)): raise ValueError("Feature matrix contains NaN or Inf.")
    return Z, names

# ============================================================
# 6. Main Pipeline Execution
# ============================================================
def main():
    if len(sys.argv) != 4: sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")
    train_path, test_path, predictions_path = sys.argv[1], sys.argv[2], sys.argv[3]
    total_start = time.perf_counter()

    print_header("PART (C) — TRAINING PIPELINE")

    # --- Step 1: Loading ---
    start = time.perf_counter()
    print_step(1, 6, "Loading raw training dataset...")
    train_df = pd.read_csv(train_path)
    feature_columns = [c for c in train_df.columns if c != "hr"]
    if len(feature_columns) != EXPECTED_RAW_FEATURES: 
        raise ValueError(f"Expected {EXPECTED_RAW_FEATURES} raw features, got {len(feature_columns)}")
    
    y_train = train_df["hr"].to_numpy(dtype=np.float64)
    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    del train_df
    
    print_stat("Samples Loaded", f"{len(y_train):,}")
    print_stat("Raw Features", X_train_raw.shape[1])
    print_time(start)

    # --- Step 2: Extracting ---
    start = time.perf_counter()
    print_step(2, 6, "Extracting physiological blueprint & frequency equalizer...")
    Z_train, feature_names = extract_features(X_train_raw, feature_columns)
    del X_train_raw
    
    print_stat("Matrix Shape", f"{Z_train.shape[0]:,} x {Z_train.shape[1]:,}")
    print_time(start)

    # --- Step 3: Standardizing (Lasso Removed) ---
    start = time.perf_counter()
    print_step(3, 6, "Standardizing Feature Matrix...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    del Z_train
    print_stat("Features Retained", Z_train_scaled.shape[1])
    print_time(start)

    # --- Step 4: ElasticNet Tuning ---
    start = time.perf_counter()
    print_step(4, 6, "Tuning SGDRegressor (ElasticNet) via 5-Fold CV...")

    # Adjusted parameters for handling dense feature space without pre-selection
    param_grid = {
        "alpha": [1e-4, 1e-3, 1e-2], 
        "epsilon": [0.01, 0.05, 0.1],
        "l1_ratio": [0.15, 0.30]
    }

    grid_search = GridSearchCV(
        estimator=SGDRegressor(loss='epsilon_insensitive', penalty='elasticnet', max_iter=2000, random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error", 
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), 
        n_jobs=1
    )
    grid_search.fit(Z_train_scaled, y_train)
    model, best_cv_mae = grid_search.best_estimator_, -grid_search.best_score_
    
    print_stat("Best Params", f"alpha={grid_search.best_params_['alpha']}, epsilon={grid_search.best_params_['epsilon']}, l1_ratio={grid_search.best_params_['l1_ratio']}")
    print_stat("Best CV MAE", f"{best_cv_mae:.4f}")
    print_time(start)

    # --- Training Metrics ---
    train_preds = model.predict(Z_train_scaled)
    train_nmae = np.sum(np.abs(y_train - train_preds)) / np.sum(np.abs(y_train - np.mean(y_train)))
    train_nmse = np.sum((y_train - train_preds) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    del train_preds, y_train, Z_train_scaled

    # --- Step 5: Test Data ---
    print_header("PART (C) — TEST PREDICTIONS")
    start = time.perf_counter()
    print_step(5, 6, "Loading and formatting test set...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    Z_test, _ = extract_features(X_test_raw, feature_columns)
    del X_test_raw

    # Direct transform (No Lasso selector needed)
    Z_test_scaled = scaler.transform(Z_test)
    del Z_test
    
    print_stat("Test Samples", f"{Z_test_scaled.shape[0]:,}")
    print_time(start)

    # --- Step 6: Prediction ---
    start = time.perf_counter()
    print_step(6, 6, "Executing model predictions...")
    predictions = model.predict(Z_test_scaled)
    if not np.all(np.isfinite(predictions)): raise ValueError("Predictions contain NaN/Inf.")
    
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print_stat("Predictions Saved To", predictions_path)
    print_time(start)

    # --- Summary ---
    print_header("FINAL SUMMARY")
    print_stat("Algorithm", "SGDRegressor (ElasticNet + Bio-Blueprint)")
    print_stat("Training NMAE", f"{train_nmae:.4f}")
    print_stat("Training NMSE", f"{train_nmse:.4f}")
    print_stat("Total Runtime", f"{time.perf_counter() - total_start:.2f}s")
    print("\nNote: Official Public/Private NMAE will be calculated by the grading script.\n" + "=" * 60)

if __name__ == "__main__":
    main()