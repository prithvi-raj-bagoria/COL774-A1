import sys
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, KFold

# ============================================================
# 1. Configuration & Hyperparameters
# ============================================================
EXPECTED_RAW_FEATURES = 1640
LASSO_ALPHA = 0.005
LASSO_MAX_ITER = 5000
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
def row_energy(x): return np.sum(x * x, axis=1, dtype=np.float32)
def row_log_energy(x): return np.log1p(np.sum(x * x, axis=1, dtype=np.float32))
def row_mad(x): return np.mean(np.abs(x - np.mean(x, axis=1, keepdims=True)), axis=1, dtype=np.float32)

def row_rmssd(x):
    """Heart Rate Variability proxy: Root mean square of successive differences."""
    return np.sqrt(np.mean(np.diff(x, axis=1)**2, axis=1, dtype=np.float32))

def row_sd1(x):
    """Poincare Plot Minor Axis (Short-term HRV)."""
    return (np.std(np.diff(x, axis=1), axis=1, dtype=np.float32) / np.sqrt(2))

def row_sd2(x):
    """Poincare Plot Major Axis (Long-term HRV)."""
    return (np.std(x[:, :-1] + x[:, 1:], axis=1, dtype=np.float32) / np.sqrt(2))

def row_sma(x, y, z):
    """Signal Magnitude Area: Actigraphy standard for physical exertion."""
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)

def row_slope(x):
    t = np.arange(x.shape[1], dtype=np.float32) - np.mean(np.arange(x.shape[1], dtype=np.float32))
    return (x @ t) / np.sum(t * t)

def row_trend_ratio(x, fraction=0.2):
    """Ratio of late window activity to early window activity (Acceleration/Deceleration)."""
    k = max(1, int(x.shape[1] * fraction))
    early = np.mean(np.abs(x[:, :k]), axis=1, dtype=np.float32) + 1e-7
    late = np.mean(np.abs(x[:, -k:]), axis=1, dtype=np.float32)
    return late / early

def row_skewness(x):
    mean, std = np.mean(x, axis=1, keepdims=True, dtype=np.float32), np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - mean) / std) ** 3, axis=1, dtype=np.float32)

def row_kurtosis(x):
    mean, std = np.mean(x, axis=1, keepdims=True, dtype=np.float32), np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - mean) / std) ** 4, axis=1, dtype=np.float32)

def row_shape_factor(x):
    """Distribution tail-heaviness (90-10 range / 75-25 range)."""
    p90, p75, p25, p10 = np.percentile(x, [90, 75, 25, 10], axis=1)
    return (p90 - p10) / (p75 - p25 + 1e-7)

def row_tkeo_mean(x):
    if x.shape[1] < 3: return np.zeros(x.shape[0], dtype=np.float32)
    tkeo = x[:, 1:-1]**2 - (x[:, 2:] * x[:, :-2])
    return np.mean(tkeo, axis=1, dtype=np.float32)

def row_hjorth_mobility(x):
    var_x = np.var(x, axis=1) + 1e-7
    var_dx = np.var(np.diff(x, axis=1), axis=1)
    return np.sqrt(var_dx / var_x).astype(np.float32)

def row_hjorth_complexity(x):
    dx = np.diff(x, axis=1)
    mob_x = row_hjorth_mobility(x) + 1e-7
    mob_dx = row_hjorth_mobility(dx)
    return (mob_dx / mob_x).astype(np.float32)

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

# ============================================================
# 4. Feature Assembly Framework
# ============================================================
def add_feat(features, names, values, name):
    features.append(np.asarray(values, dtype=np.float32))
    names.append(name)

def add_basic_features(features, names, x, prefix):
    for val, suffix in zip(
        [row_mean(x), row_std(x), row_min(x), row_max(x), row_range(x), row_rms(x), row_slope(x), row_mad(x), row_energy(x), row_log_energy(x), row_rmssd(x), row_sd1(x), row_sd2(x), row_trend_ratio(x)],
        ["mean", "std", "min", "max", "range", "rms", "slope", "mad", "energy", "log_energy", "rmssd", "sd1", "sd2", "trend_ratio"]
    ): add_feat(features, names, val, f"{prefix}_{suffix}")

def add_deep_features(features, names, x, prefix, autocorr_lags=None):
    add_basic_features(features, names, x, prefix)
    sorted_x, n = np.sort(x, axis=1), x.shape[1]
    
    for p, suffix in zip([0.05, 0.25, 0.75, 0.95], ["p05", "p25", "p75", "p95"]):
        add_feat(features, names, sorted_x[:, int(p * (n - 1))], f"{prefix}_{suffix}")
        
    add_feat(features, names, row_skewness(x), f"{prefix}_skew")
    add_feat(features, names, row_kurtosis(x), f"{prefix}_kurt")
    add_feat(features, names, row_shape_factor(x), f"{prefix}_shape_factor")
    add_feat(features, names, row_tkeo_mean(x), f"{prefix}_tkeo")
    add_feat(features, names, row_hjorth_mobility(x), f"{prefix}_hjorth_mob")
    add_feat(features, names, row_hjorth_complexity(x), f"{prefix}_hjorth_comp")
    
    if autocorr_lags:
        for lag in autocorr_lags: add_feat(features, names, row_autocorr_lag(x, lag), f"{prefix}_autocorr_{lag}")
        
    add_feat(features, names, row_zero_crossings(x), f"{prefix}_zero_cross")
    add_feat(features, names, row_local_extrema(x), f"{prefix}_extrema")

def add_temporal_blocks(features, names, x, prefix, n_blocks=4):
    bs = x.shape[1] // n_blocks
    for k in range(n_blocks):
        block = x[:, k * bs:(k + 1) * bs]
        add_feat(features, names, row_mean(block), f"{prefix}_t{k + 1}_mean")
        add_feat(features, names, row_std(block), f"{prefix}_t{k + 1}_std")
        add_feat(features, names, row_slope(block), f"{prefix}_t{k + 1}_slope")
        add_feat(features, names, row_rmssd(block), f"{prefix}_t{k + 1}_rmssd")

# ============================================================
# 5. Core Feature Extraction
# ============================================================
def extract_features(X_raw, feature_columns):
    features, names = [], []
    
    acc_x = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]]
    acc_y = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]]
    acc_z = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]]
    bvp = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]]
    eda = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]]

    # --- Cardiac Features (BVP, VPG, APG) ---
    add_deep_features(features, names, bvp, "bvp", [16, 24, 32, 44, 58, 76, 96])
    add_temporal_blocks(features, names, bvp, "bvp")

    vpg, apg = np.diff(bvp, axis=1), np.diff(np.diff(bvp, axis=1), axis=1)
    add_deep_features(features, names, vpg, "vpg")
    add_deep_features(features, names, apg, "apg")

    estimated_bpm = (np.sum((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0), axis=1) * 6.0).astype(np.float32)
    add_feat(features, names, estimated_bpm, "vpg_estimated_bpm")

    # --- Motion Features (Accelerometer & Jerk) ---
    add_basic_features(features, names, acc_x, "acc_x")
    add_basic_features(features, names, acc_y, "acc_y")
    add_basic_features(features, names, acc_z, "acc_z")

    add_feat(features, names, row_sma(acc_x, acc_y, acc_z), "acc_sma")

    acc_sq = acc_x ** 2 + acc_y ** 2 + acc_z ** 2
    add_deep_features(features, names, acc_sq, "acc_sq", [1, 2, 4, 8])
    add_temporal_blocks(features, names, acc_sq, "acc_sq")
    
    jerk = np.diff(acc_sq, axis=1)
    add_deep_features(features, names, jerk, "jerk")

    # --- Stress/Electrodermal Features (EDA + Phasic EDA) ---
    # Log transform EDA (sweat often scales logarithmically)
    eda_log = np.log1p(np.clip(eda, 0, None))
    add_deep_features(features, names, eda, "eda")
    add_basic_features(features, names, eda_log, "eda_log")
    add_temporal_blocks(features, names, eda, "eda")
    
    phasic_eda = np.diff(eda, axis=1)
    add_basic_features(features, names, phasic_eda, "phasic_eda")

    # --- NON-LINEAR POLYNOMIALS & CROSS-MODAL PROXIES ---
    acc_rms_val = row_rms(acc_sq)
    eda_mean_val = row_mean(eda)
    bvp_std_val = row_std(bvp)
    
    # 2nd Order Terms (Curve fitting)
    add_feat(features, names, estimated_bpm ** 2, "bpm_squared")
    add_feat(features, names, acc_rms_val ** 2, "acc_rms_squared")
    add_feat(features, names, eda_mean_val ** 2, "eda_mean_squared")
    
    # Interactions
    add_feat(features, names, estimated_bpm * acc_rms_val, "inter_bpm_motion")
    add_feat(features, names, estimated_bpm * np.log1p(acc_rms_val), "inter_bpm_log_motion")
    add_feat(features, names, bvp_std_val * row_std(acc_sq), "inter_bvp_motion_var")
    add_feat(features, names, estimated_bpm * eda_mean_val, "inter_bpm_eda")
    add_feat(features, names, acc_rms_val * eda_mean_val, "inter_motion_eda")
    
    # Stress vs Exercise Ratio
    add_feat(features, names, eda_mean_val / (acc_rms_val + 1e-5), "ratio_eda_motion")

    # SNR Proxy
    snr = np.var(bvp, axis=1) / (np.var(acc_sq, axis=1) + 1e-5)
    add_feat(features, names, snr, "time_domain_snr")
    add_feat(features, names, np.log1p(snr), "time_domain_snr_log")

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

    start = time.perf_counter()
    print_step(2, 6, "Extracting biological, non-linear & physical features...")
    Z_train, feature_names = extract_features(X_train_raw, feature_columns)
    del X_train_raw
    
    print_stat("Matrix Shape", f"{Z_train.shape[0]:,} x {Z_train.shape[1]:,}")
    print_time(start)

    start = time.perf_counter()
    print_step(3, 6, f"Standardizing & applying Lasso Selection (alpha={LASSO_ALPHA})...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    del Z_train
    
    selector = SelectFromModel(
        Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER, random_state=RANDOM_STATE, tol=0.001), 
        prefit=False
    )
    Z_train_selected = selector.fit_transform(Z_train_scaled, y_train)
    del Z_train_scaled
    
    print_stat("Features Retained", Z_train_selected.shape[1])
    print_stat("Features Pruned", len(feature_names) - Z_train_selected.shape[1])
    print_time(start)

    start = time.perf_counter()
    print_step(4, 6, "Tuning SGDRegressor (epsilon-insensitive) via 5-Fold CV...")

    # Expanded alpha to accommodate the highly non-linear interaction terms
    param_grid = {
        "alpha": [1e-5, 1e-4, 1e-3], 
        "epsilon": [0.01, 0.05, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=SGDRegressor(loss='epsilon_insensitive', penalty='l2', max_iter=2000, random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error", 
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), 
        n_jobs=1
    )
    grid_search.fit(Z_train_selected, y_train)
    model, best_cv_mae = grid_search.best_estimator_, -grid_search.best_score_
    
    print_stat("Best Params", f"alpha={grid_search.best_params_['alpha']}, epsilon={grid_search.best_params_['epsilon']}")
    print_stat("Best CV MAE", f"{best_cv_mae:.4f}")
    print_time(start)

    train_preds = model.predict(Z_train_selected)
    train_nmae = np.sum(np.abs(y_train - train_preds)) / np.sum(np.abs(y_train - np.mean(y_train)))
    train_nmse = np.sum((y_train - train_preds) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    del train_preds, y_train, Z_train_selected

    print_header("PART (C) — TEST PREDICTIONS")
    start = time.perf_counter()
    print_step(5, 6, "Loading and formatting test set...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    Z_test, _ = extract_features(X_test_raw, feature_columns)
    del X_test_raw

    Z_test_selected = selector.transform(scaler.transform(Z_test))
    del Z_test
    
    print_stat("Test Samples", f"{Z_test_selected.shape[0]:,}")
    print_time(start)

    start = time.perf_counter()
    print_step(6, 6, "Executing model predictions...")
    predictions = model.predict(Z_test_selected)
    if not np.all(np.isfinite(predictions)): raise ValueError("Predictions contain NaN/Inf.")
    
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print_stat("Predictions Saved To", predictions_path)
    print_time(start)

    print_header("FINAL SUMMARY")
    print_stat("Algorithm", "SGDRegressor (Non-Linear Bio-Polynomials + Lasso)")
    print_stat("Training NMAE", f"{train_nmae:.4f}")
    print_stat("Training NMSE", f"{train_nmse:.4f}")
    print_stat("Total Runtime", f"{time.perf_counter() - total_start:.2f}s")
    print("\nNote: Official Public/Private NMAE will be calculated by the grading script.\n" + "=" * 60)

if __name__ == "__main__":
    main()