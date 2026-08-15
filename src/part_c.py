%%writefile src/part_c.py
import sys
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Configuration
# ============================================================
EXPECTED_RAW_FEATURES = 1640
LASSO_MAX_FEATURES = 500          # optimal balance for polynomial SGD
RANDOM_STATE = 42

# Lasso selector on subsample
LASSO_ALPHA = 0.001
LASSO_MAX_ITER = 3000
LASSO_TOL = 1e-4
SUBSAMPLE_SIZE = 100_000            # rows used for Lasso feature selection

# SGDRegressor (Linear MAE Optimization) Tuning Grid
CV_FOLDS = 3
SGD_ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-2, 1.0]

# ============================================================
# Output helpers
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
# Fast vectorized signal math
# ============================================================
def row_mean(x): return np.mean(x, axis=1, dtype=np.float32)
def row_std(x): return np.std(x, axis=1, dtype=np.float32)

def row_skewness(x):
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - m) / s) ** 3, axis=1, dtype=np.float32)

def row_kurtosis(x):
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - m) / s) ** 4, axis=1, dtype=np.float32)

def row_autocorr_lag(x, lag):
    if x.shape[1] <= lag:
        return np.zeros(x.shape[0], dtype=np.float32)
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return (np.sum(c[:, :-lag] * c[:, lag:], axis=1) /
            (np.sum(c * c, axis=1) + 1e-10)).astype(np.float32)

def row_zero_crossings(x):
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum((c[:, :-1] * c[:, 1:]) < 0, axis=1).astype(np.float32)

def row_local_extrema(x):
    d = np.diff(x, axis=1)
    return np.sum((d[:, :-1] * d[:, 1:]) < 0, axis=1).astype(np.float32)

def row_sma(x, y, z):
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)

def row_tkeo_mean(x):
    if x.shape[1] < 3:
        return np.zeros(x.shape[0], dtype=np.float32)
    tkeo = x[:, 1:-1]**2 - (x[:, :-2] * x[:, 2:])
    return np.mean(tkeo, axis=1, dtype=np.float32)

def row_shannon_entropy(x):
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - m) / s
    P = np.column_stack([
        np.mean(z < -2, axis=1),
        np.mean((z >= -2) & (z < -1), axis=1),
        np.mean((z >= -1) & (z < 0), axis=1),
        np.mean((z >= 0) & (z < 1), axis=1),
        np.mean((z >= 1) & (z < 2), axis=1),
        np.mean(z >= 2, axis=1)
    ]) + 1e-10
    return -np.sum(P * np.log(P), axis=1).astype(np.float32)

def row_mad(x):
    m = np.mean(x, axis=1, keepdims=True)
    return np.mean(np.abs(x - m), axis=1, dtype=np.float32)

def row_rms(x):
    return np.sqrt(np.mean(x**2, axis=1, dtype=np.float32))

def row_variance(x):
    return np.var(x, axis=1, dtype=np.float32)

def row_roll_pitch(acc_x, acc_y, acc_z):
    roll = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2) + 1e-7)
    pitch = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2) + 1e-7)
    return roll, pitch

def row_cross_corr(x, y):
    xm = x - np.mean(x, axis=1, keepdims=True)
    ym = y - np.mean(y, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(xm**2, axis=1) * np.sum(ym**2, axis=1)) + 1e-10
    return np.sum(xm * ym, axis=1) / denom

def row_dominant_axis_ratio(ax, ay, az):
    var_x = np.var(ax, axis=1)
    var_y = np.var(ay, axis=1)
    var_z = np.var(az, axis=1)
    return np.maximum.reduce([var_x, var_y, var_z]) / (var_x + var_y + var_z + 1e-10)

def row_postural_transitions(acc_x, acc_y, acc_z):
    ax_abs = np.abs(acc_x)
    ay_abs = np.abs(acc_y)
    az_abs = np.abs(acc_z)
    dominant = np.argmax(np.stack([ax_abs, ay_abs, az_abs], axis=2), axis=2)
    transitions = np.sum(dominant[:, :-1] != dominant[:, 1:], axis=1)
    return transitions.astype(np.float32)

def row_percentile(x, q):
    return np.percentile(x, q, axis=1).astype(np.float32)

def row_turning_point_ratio(x):
    return row_local_extrema(x) / x.shape[1]

# ============================================================
# Feature extraction
# ============================================================
def extract_base_features(X_raw, feature_columns):
    features, names = [], []

    acc_x = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]]
    acc_y = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]]
    acc_z = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]]
    bvp = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]]
    eda = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]]

    vpg = np.diff(bvp, axis=1)
    apg = np.diff(vpg, axis=1)
    acc_sq = acc_x**2 + acc_y**2 + acc_z**2
    acc_mag = np.sqrt(acc_sq)

    # --------------------------------------------------------
    # I. BLOOD VOLUME PULSE (BVP) FEATURES
    # --------------------------------------------------------
    # 1. Pulse Morphology
    features.append(row_skewness(bvp)); names.append("bvp_skew")
    features.append(row_kurtosis(bvp)); names.append("bvp_kurt")
    features.append(row_mad(bvp)); names.append("bvp_mad")
    features.append(row_percentile(bvp, 75) - row_percentile(bvp, 25)); names.append("bvp_iqr")
    features.append(np.max(bvp, axis=1) - np.min(bvp, axis=1)); names.append("bvp_peak2peak")
    features.append(row_percentile(bvp, 90) - row_percentile(bvp, 10)); names.append("bvp_p90_p10")
    features.append(np.max(np.abs(bvp), axis=1) / (row_rms(bvp) + 1e-7)); names.append("bvp_crest_factor")
    features.append(row_rms(bvp)); names.append("bvp_rms")
    features.append(row_mean(bvp)); names.append("bvp_mean")

    # 2. VPG (Velocity)
    features.append(row_mean(np.abs(vpg))); names.append("vpg_mean_abs")
    features.append(row_variance(vpg)); names.append("vpg_var")
    features.append(row_skewness(vpg)); names.append("vpg_skew")
    features.append(row_kurtosis(vpg)); names.append("vpg_kurt")
    features.append(np.max(vpg, axis=1)); names.append("vpg_max")
    features.append(np.min(vpg, axis=1)); names.append("vpg_min")
    features.append(row_zero_crossings(vpg)); names.append("vpg_zcross")
    features.append(np.sum(np.abs(np.diff(vpg, axis=1)), axis=1)); names.append("vpg_line_length")

    # 3. APG (Acceleration)
    features.append(row_variance(apg)); names.append("apg_var")
    features.append(row_mean(np.abs(apg))); names.append("apg_mean_abs")
    features.append(row_skewness(apg)); names.append("apg_skew")
    features.append(row_kurtosis(apg)); names.append("apg_kurt")
    features.append(row_zero_crossings(apg)); names.append("apg_zcross")
    features.append(row_turning_point_ratio(apg)); names.append("apg_turning_ratio")

    # 4. Autocorrelation Equalizer
    ac32 = row_autocorr_lag(bvp, 32)
    ac38 = row_autocorr_lag(bvp, 38)
    ac48 = row_autocorr_lag(bvp, 48)
    ac64 = row_autocorr_lag(bvp, 64)
    ac77 = row_autocorr_lag(bvp, 77)
    features.append(ac32); names.append("bvp_ac_32")
    features.append(ac38); names.append("bvp_ac_38")
    features.append(ac48); names.append("bvp_ac_48")
    features.append(ac64); names.append("bvp_ac_64")
    features.append(ac77); names.append("bvp_ac_77")
    features.append((ac32 + ac38) / (ac64 + ac77 + 1e-7)); names.append("bvp_ac_high_low_ratio")

    # 5. Signal Quality Index (SQI)
    features.append(row_shannon_entropy(bvp)); names.append("bvp_entropy")
    features.append(row_tkeo_mean(bvp)); names.append("bvp_tkeo")
    bvp_std, vpg_std, apg_std = row_std(bvp) + 1e-7, row_std(vpg) + 1e-7, row_std(apg) + 1e-7
    hj_mob = vpg_std / bvp_std
    features.append(hj_mob); names.append("bvp_hjorth_mob")
    features.append((apg_std / vpg_std) / hj_mob); names.append("bvp_hjorth_com")


    # --------------------------------------------------------
    # II. ACCELEROMETER FEATURES
    # --------------------------------------------------------
    # 1. Kinematics & Energy
    features.append(row_sma(acc_x, acc_y, acc_z)); names.append("acc_sma")
    features.append(row_mad(acc_mag)); names.append("acc_mad")
    features.append(row_variance(acc_sq)); names.append("acc_norm_sq_var")
    features.append(row_rms(acc_mag)); names.append("acc_rms")
    features.append(row_tkeo_mean(acc_x) + row_tkeo_mean(acc_y) + row_tkeo_mean(acc_z)); names.append("acc_tkeo")
    jerk_mag = np.sqrt(np.diff(acc_x, axis=1)**2 + np.diff(acc_y, axis=1)**2 + np.diff(acc_z, axis=1)**2)
    features.append(row_mean(jerk_mag)); names.append("acc_jerk_mean_abs")
    snap_mag = np.sqrt(np.diff(acc_x, n=2, axis=1)**2 + np.diff(acc_y, n=2, axis=1)**2 + np.diff(acc_z, n=2, axis=1)**2)
    features.append(row_mean(snap_mag)); names.append("acc_snap_mean_abs")
    features.append(row_std(acc_mag) / (row_mean(acc_mag) + 1e-7)); names.append("acc_cv")
    features.append(np.sum(np.abs(acc_mag - row_mean(acc_mag).reshape(-1, 1)), axis=1).astype(np.float32)); names.append("acc_vib_int")

    # 2. Spatial Biomechanics & Posture
    features.append(row_mean(acc_x)); names.append("acc_mean_x")
    features.append(row_mean(acc_y)); names.append("acc_mean_y")
    features.append(row_mean(acc_z)); names.append("acc_mean_z")
    roll, pitch = row_roll_pitch(acc_x, acc_y, acc_z)
    features.append(row_mean(roll)); names.append("acc_roll")
    features.append(row_mean(pitch)); names.append("acc_pitch")
    features.append(row_cross_corr(acc_x, acc_y)); names.append("acc_corr_xy")
    features.append(row_cross_corr(acc_x, acc_z)); names.append("acc_corr_xz")
    features.append(row_cross_corr(acc_y, acc_z)); names.append("acc_corr_yz")
    features.append(row_dominant_axis_ratio(acc_x, acc_y, acc_z)); names.append("acc_dom_axis_ratio")
    features.append(row_postural_transitions(acc_x, acc_y, acc_z)); names.append("acc_post_transitions")

    # 3. Probability Distribution (Impacts)
    features.append(row_skewness(acc_mag)); names.append("acc_skew")
    features.append(row_kurtosis(acc_mag)); names.append("acc_kurt")
    features.append(row_percentile(acc_mag, 75) - row_percentile(acc_mag, 25)); names.append("acc_iqr")
    features.append(np.max(acc_mag, axis=1) / (row_rms(acc_mag) + 1e-7)); names.append("acc_crest_factor")
    features.append(np.max(acc_mag, axis=1) - np.min(acc_mag, axis=1)); names.append("acc_peak2peak")
    features.append(row_percentile(acc_mag, 10)); names.append("acc_p10")
    features.append(row_percentile(acc_mag, 90)); names.append("acc_p90")

    # 4. Frequency Proxies
    features.append(row_zero_crossings(acc_mag)); names.append("acc_zcross")
    features.append(row_turning_point_ratio(acc_mag)); names.append("acc_turning_ratio")
    diff_mag = np.diff(acc_mag, axis=1)
    diff2_mag = np.diff(diff_mag, axis=1)
    hj_mob_acc = row_std(diff_mag) / (row_std(acc_mag) + 1e-7)
    features.append(hj_mob_acc); names.append("acc_hjorth_mob")
    features.append((row_std(diff2_mag) / (row_std(diff_mag) + 1e-7)) / (hj_mob_acc + 1e-7)); names.append("acc_hjorth_com")
    features.append(row_autocorr_lag(acc_mag, 16)); names.append("acc_ac_16")
    features.append(row_autocorr_lag(acc_mag, 32)); names.append("acc_ac_32")
    features.append(row_shannon_entropy(acc_mag)); names.append("acc_entropy")

    # --------------------------------------------------------
    # III. ELECTRODERMAL ACTIVITY (EDA) FEATURES
    # --------------------------------------------------------
    # 1. Tonic Baseline Envelopes
    features.append(row_mean(eda)); names.append("eda_mean")
    features.append(row_std(eda)); names.append("eda_std")
    features.append(np.min(eda, axis=1)); names.append("eda_min")
    features.append(np.max(eda, axis=1)); names.append("eda_max")
    features.append(row_percentile(eda, 75) - row_percentile(eda, 25)); names.append("eda_iqr")
    
    # 2. Dynamic Trend (Slope Proxy)
    eda_start_mean = np.mean(eda[:, :10], axis=1)
    eda_end_mean = np.mean(eda[:, -10:], axis=1)
    features.append(eda_end_mean - eda_start_mean); names.append("eda_slope_proxy")
    
    # 3. High-Frequency Neural Jitter
    features.append(np.sum(np.abs(np.diff(eda, axis=1)), axis=1)); names.append("eda_line_length")
    features.append(row_zero_crossings(np.diff(eda, axis=1))); names.append("eda_diff_zcross")

    # 4. Phasic Skin Conductance Response (Spikes)
    window = 8
    cs = np.cumsum(eda, axis=1, dtype=np.float32)
    tonic = (cs[:, window:] - cs[:, :-window]) / window
    phasic = eda[:, window:] - tonic
    
    features.append(np.sum(phasic**2, axis=1)); names.append("eda_phasic_energy")
    features.append(np.max(phasic, axis=1)); names.append("eda_phasic_max")
    features.append(np.std(phasic, axis=1)); names.append("eda_phasic_std")
    
    # 5. Peak Impact & Morphological Shape
    features.append(row_skewness(eda)); names.append("eda_skew")
    features.append(row_kurtosis(eda)); names.append("eda_kurt")
    features.append((np.argmax(eda, axis=1) / 40.0).astype(np.float32)); names.append("eda_peak_time")
    features.append(row_rms(eda)); names.append("eda_rms")

    Z_base = np.column_stack(features).astype(np.float32)
    return Z_base, names

def expand_polynomials(Z_base, names_base):
    n, p = Z_base.shape
    count = p + p * (p + 1) // 2
    out = np.empty((n, count), dtype=np.float32)
    out_names = list(names_base)
    out[:, :p] = Z_base
    col = p
    for i in range(p):
        zi = Z_base[:, i]
        for j in range(i, p):
            out[:, col] = zi * Z_base[:, j]
            out_names.append(f"{names_base[i]}^2" if i == j else f"{names_base[i]}*{names_base[j]}")
            col += 1
    if not np.all(np.isfinite(out)):
        raise ValueError("Polynomial matrix contains NaN/Inf.")
    return out, out_names

def lasso_select_subsample(X_scaled, y, max_features, subsample_size):
    """
    Feature selection using Lasso on a random subsample.
    """
    np.random.seed(RANDOM_STATE)
    if X_scaled.shape[0] > subsample_size:
        idx = np.random.choice(X_scaled.shape[0], subsample_size, replace=False)
        X_sub = X_scaled[idx]
        y_sub = y[idx]
    else:
        X_sub = X_scaled
        y_sub = y

    lasso = Lasso(
        alpha=LASSO_ALPHA,
        max_iter=LASSO_MAX_ITER,
        tol=LASSO_TOL,
        random_state=RANDOM_STATE,
        precompute=True
    )
    lasso.fit(X_sub, y_sub)

    coef_abs = np.abs(lasso.coef_)
    n_features = X_scaled.shape[1]
    if n_features > max_features:
        selected_idx = np.argsort(coef_abs)[-max_features:]
        selected_idx = np.sort(selected_idx)
    else:
        selected_idx = np.arange(n_features)
    return selected_idx

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")
    
    train_path, test_path, predictions_path = sys.argv[1], sys.argv[2], sys.argv[3]
    total_start = time.perf_counter()
    print_header("PART (C) — OPTIMIZED PIPELINE (SGDREGRESSOR NMAE)")

    # Step 1
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

    # Step 2
    start = time.perf_counter()
    print_step(2, 6, "Building Base & Polynomial Feature Matrix...")
    Z_train_base, base_names = extract_base_features(X_train_raw, feature_columns)
    del X_train_raw
    Z_train_poly, poly_names = expand_polynomials(Z_train_base, base_names)
    del Z_train_base
    print_stat("Base Features", len(base_names))
    print_stat("Expanded Poly Features", Z_train_poly.shape[1])
    print_time(start)

    # Step 3
    start = time.perf_counter()
    print_step(3, 6, "Standardizing & Lasso feature selection (subsampled)...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train_poly)
    del Z_train_poly
    lasso_idx = lasso_select_subsample(Z_train_scaled, y_train, LASSO_MAX_FEATURES, SUBSAMPLE_SIZE)
    Z_train_selected = Z_train_scaled[:, lasso_idx]
    del Z_train_scaled
    print_stat("Features Retained", Z_train_selected.shape[1])
    print_stat("Features Pruned", len(poly_names) - Z_train_selected.shape[1])
    print_time(start)

    # Step 4
    start = time.perf_counter()
    print_step(4, 6, "Tuning SGDRegressor (MAE Loss) via Cross-Validation...")
    
    # SGDRegressor natively handles L1 'epsilon_insensitive' loss on huge datasets
    grid_search = GridSearchCV(
        estimator=SGDRegressor(
            loss='epsilon_insensitive', 
            epsilon=0.0, 
            penalty='l2',
            learning_rate='adaptive', # Dynamically adjusts step size for stable convergence
            eta0=0.01,
            random_state=RANDOM_STATE,
            max_iter=5000,
            tol=1e-4
        ),
        param_grid={"alpha": SGD_ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    grid_search.fit(Z_train_selected, y_train)
    model = grid_search.best_estimator_
    best_cv_mae = -grid_search.best_score_

    print_stat("Best SGD Alpha", f"{grid_search.best_params_['alpha']}")
    print_stat("Best CV MAE", f"{best_cv_mae:.4f}")

    train_preds = model.predict(Z_train_selected)
    train_nmae = np.sum(np.abs(y_train - train_preds)) / np.sum(np.abs(y_train - np.mean(y_train)))
    train_nmse = np.sum((y_train - train_preds) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    print_stat("Training NMAE", f"{train_nmae:.4f}")
    print_stat("Training NMSE", f"{train_nmse:.4f}")
    print_time(start)

    # Step 5
    print_header("PART (C) — TEST PREDICTIONS")
    start = time.perf_counter()
    print_step(5, 6, "Processing test set...")
    test_df = pd.read_csv(test_path)
    y_test = test_df["hr"].to_numpy(dtype=np.float64) if "hr" in test_df.columns else None
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df
    
    Z_test_base, _ = extract_base_features(X_test_raw, feature_columns)
    del X_test_raw
    Z_test_poly, _ = expand_polynomials(Z_test_base, base_names)
    del Z_test_base
    
    Z_test_scaled = scaler.transform(Z_test_poly)
    Z_test_selected = Z_test_scaled[:, lasso_idx]
    del Z_test_poly, Z_test_scaled
    print_stat("Test Samples", f"{Z_test_selected.shape[0]:,}")
    print_time(start)

    # Step 6
    start = time.perf_counter()
    print_step(6, 6, "Generating predictions...")
    predictions = model.predict(Z_test_selected)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions contain NaN/Inf.")
    
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print_stat("Predictions Saved To", predictions_path)
    
    test_nmae, test_nmse = None, None
    if y_test is not None:
        test_nmae = np.sum(np.abs(y_test - predictions)) / np.sum(np.abs(y_test - np.mean(y_test)))
        test_nmse = np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print_time(start)

    # Summary
    print_header("FINAL SUMMARY")
    print_stat("Algorithm", "SGDRegressor (MAE) + Lasso Subsampled Selection")
    print_stat("Training NMAE", f"{train_nmae:.4f}")
    print_stat("Training NMSE", f"{train_nmse:.4f}")
    if test_nmae is not None:
        print_stat("Test NMAE", f"{test_nmae:.4f}")
        print_stat("Test NMSE", f"{test_nmse:.4f}")
    print_stat("Total Runtime", f"{time.perf_counter() - total_start:.2f}s")
    print("\nNote: Official Public/Private NMAE will be calculated by the grading script.\n" + "=" * 60)

if __name__ == "__main__":
    main()