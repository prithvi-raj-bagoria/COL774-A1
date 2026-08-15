import sys
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import ConvergenceWarning

# Suppress Lasso convergence warnings to keep output clean
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# 1. Configuration & Hyperparameters
# ============================================================
EXPECTED_RAW_FEATURES = 1640
LASSO_ALPHA = 0.005
LASSO_MAX_ITER = 1000
LASSO_TOL = 1e-5
LASSO_MAX_FEATURES = 500          # manually select this many features after Lasso
CV_FOLDS = 3
RANDOM_STATE = 42

# SGD alpha grid for final hyperparameter tuning (MAE loss)
SGD_ALPHAS = [1e-5, 1e-4, 1e-3]

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
    """Signal Magnitude Area: Actigraphy standard for physical exertion."""
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)

def row_tkeo_mean(x):
    """Teager-Kaiser Energy Operator: Penalizes rapid, erratic motion shocks."""
    if x.shape[1] < 3:
        return np.zeros(x.shape[0], dtype=np.float32)
    tkeo = x[:, 1:-1]**2 - (x[:, :-2] * x[:, 2:])
    return np.mean(tkeo, axis=1, dtype=np.float32)

def row_shannon_entropy(x):
    """Fast Vectorized Shannon Entropy Proxy for Signal Quality Index (SQI)."""
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

def row_eda_phasic_energy(x):
    """Isolates Phasic EDA (stress spikes) from Tonic EDA (baseline drift)."""
    window = 8
    if x.shape[1] < window:
        return np.zeros(x.shape[0], dtype=np.float32)

    cs = np.cumsum(x, axis=1, dtype=np.float32)
    tonic = (cs[:, window:] - cs[:, :-window]) / window
    phasic = x[:, window:] - tonic
    return np.sum(phasic**2, axis=1).astype(np.float32)

# ============================================================
# 4. Feature Assembly Framework (Base & Polynomial)
# ============================================================
def extract_base_features(X_raw, feature_columns):
    """Extracts a tight, highly curated set of advanced biological base features."""
    features, names = [], []

    acc_x = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]]
    acc_y = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]]
    acc_z = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]]
    bvp = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]]
    eda = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]]

    # Pre-compute derivatives for multiple feature sets
    vpg = np.diff(bvp, axis=1)
    apg = np.diff(vpg, axis=1)

    # --- A. BVP Pulse Morphology & SQI ---
    features.append(row_std(bvp)); names.append("bvp_std")
    features.append(row_skewness(bvp)); names.append("bvp_skew")
    features.append(row_kurtosis(bvp)); names.append("bvp_kurt")
    features.append(row_zero_crossings(bvp)); names.append("bvp_zcross")
    features.append(row_local_extrema(bvp)); names.append("bvp_extrema")
    features.append(row_shannon_entropy(bvp)); names.append("bvp_entropy_sqi")

    # --- B. Hjorth Parameters (Waveform Complexity) ---
    bvp_std_val = row_std(bvp) + 1e-7
    vpg_std_val = row_std(vpg) + 1e-7
    apg_std_val = row_std(apg) + 1e-7

    mobility_bvp = vpg_std_val / bvp_std_val
    mobility_vpg = apg_std_val / vpg_std_val
    complexity_bvp = mobility_vpg / (mobility_bvp + 1e-7)

    features.append(mobility_bvp); names.append("bvp_hjorth_mobility")
    features.append(complexity_bvp); names.append("bvp_hjorth_complexity")

    # --- C. Frequency Equalizer via Autocorrelation Bins ---
    bpm_targets = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170]
    for bpm in bpm_targets:
        lag = int(60.0 * 64 / bpm)
        features.append(row_autocorr_lag(bvp, lag))
        names.append(f"bvp_ac_{bpm}bpm")

    # --- D. Non-FFT Frequency Proxies ---
    features.append(row_zero_crossings(vpg)); names.append("vpg_zcross_freq")
    features.append(row_zero_crossings(apg)); names.append("apg_zcross_freq")

    ac_energy_low = row_autocorr_lag(bvp, 32) + row_autocorr_lag(bvp, 38)
    ac_energy_high = row_autocorr_lag(bvp, 16) + row_autocorr_lag(bvp, 22)
    features.append(ac_energy_low); names.append("bvp_ac_energy_low")
    features.append(ac_energy_high); names.append("bvp_ac_energy_high")

    # --- E. VPG & Advanced APG Shape Statistics ---
    features.append(vpg_std_val); names.append("vpg_std")
    features.append(row_skewness(vpg)); names.append("vpg_skew")

    features.append(apg_std_val); names.append("apg_std")
    features.append(row_skewness(apg)); names.append("apg_skew")
    features.append(row_kurtosis(apg)); names.append("apg_kurt")
    features.append(row_zero_crossings(apg)); names.append("apg_zcross")
    features.append(row_local_extrema(apg)); names.append("apg_extrema")

    est_bpm = (np.sum((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0), axis=1) * 6.0).astype(np.float32)
    features.append(est_bpm); names.append("vpg_est_bpm")

    # --- F. Context: Motion & TKEO Artifact Detection ---
    acc_sma_val = row_sma(acc_x, acc_y, acc_z)
    features.append(acc_sma_val); names.append("acc_sma")

    acc_sq = acc_x**2 + acc_y**2 + acc_z**2
    features.append(row_std(acc_sq)); names.append("acc_sq_std")

    acc_tkeo_tot = row_tkeo_mean(acc_x) + row_tkeo_mean(acc_y) + row_tkeo_mean(acc_z)
    features.append(acc_tkeo_tot); names.append("acc_tkeo_total")

    # --- G. Stress: Phasic vs Tonic EDA ---
    features.append(row_mean(eda)); names.append("eda_mean")
    features.append(row_std(eda)); names.append("eda_std")
    features.append(row_std(np.diff(eda, axis=1))); names.append("eda_diff_std")
    features.append(row_eda_phasic_energy(eda)); names.append("eda_phasic_energy")

    # --- H. Temporal Context (Acceleration/Deceleration) ---
    half_bvp = bvp.shape[1] // 2
    half_acc = acc_x.shape[1] // 2

    features.append(row_std(bvp[:, :half_bvp])); names.append("bvp_std_h1")
    features.append(row_std(bvp[:, half_bvp:])); names.append("bvp_std_h2")
    features.append(row_sma(acc_x[:, :half_acc], acc_y[:, :half_acc], acc_z[:, :half_acc])); names.append("acc_sma_h1")
    features.append(row_sma(acc_x[:, half_acc:], acc_y[:, half_acc:], acc_z[:, half_acc:])); names.append("acc_sma_h2")

    Z_base = np.column_stack(features).astype(np.float32)
    return Z_base, names

def expand_polynomials(Z_base, names_base):
    """
    100% Legal Pure-NumPy Polynomial Expander.
    Generates Degree-2 interactions (squared terms + cross-multiplications).
    """
    poly_features, poly_names = [], []
    n_cols = Z_base.shape[1]

    # 1. Base Features (Degree 1)
    for i in range(n_cols):
        poly_features.append(Z_base[:, i])
        poly_names.append(names_base[i])

    # 2. Interactions & Squared (Degree 2)
    for i in range(n_cols):
        for j in range(i, n_cols):
            poly_features.append(Z_base[:, i] * Z_base[:, j])
            if i == j:
                poly_names.append(f"{names_base[i]}^2")
            else:
                poly_names.append(f"{names_base[i]}*{names_base[j]}")

    Z_poly = np.column_stack(poly_features).astype(np.float32)
    if not np.all(np.isfinite(Z_poly)):
        raise ValueError("Poly matrix contains NaN/Inf.")
    return Z_poly, poly_names

# ============================================================
# 5. Manual Lasso Feature Selection 
# ============================================================
def manual_lasso_select(X_scaled, y, max_features):
    """
    Fits a Lasso model (with precompute=True) and selects the top max_features
    features based on absolute coefficient magnitude.
    This is fully manual and transparent.
    """
    lasso = Lasso(
        alpha=LASSO_ALPHA,
        max_iter=LASSO_MAX_ITER,
        tol=LASSO_TOL,
        random_state=RANDOM_STATE,
        precompute=True
    )
    lasso.fit(X_scaled, y)

    coef_abs = np.abs(lasso.coef_)
    n_features = X_scaled.shape[1]

    if n_features > max_features:
        selected_idx = np.argsort(coef_abs)[-max_features:]
        selected_idx = np.sort(selected_idx)          # preserve original column order
    else:
        selected_idx = np.arange(n_features)

    return selected_idx

# ============================================================
# 6. Main Pipeline Execution
# ============================================================
def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")

    train_path, test_path, predictions_path = sys.argv[1], sys.argv[2], sys.argv[3]
    total_start = time.perf_counter()

    print_header("PART (C) — TRAINING PIPELINE (ADVANCED BIOSIGNALS)")

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

    # --- Step 2: Extract & Expand ---
    start = time.perf_counter()
    print_step(2, 6, "Building Base & Polynomial Feature Matrix (Pure NumPy)...")

    Z_train_base, base_names = extract_base_features(X_train_raw, feature_columns)
    del X_train_raw

    Z_train_poly, poly_names = expand_polynomials(Z_train_base, base_names)
    del Z_train_base

    print_stat("Base Features", len(base_names))
    print_stat("Expanded Poly Features", Z_train_poly.shape[1])
    print_time(start)

    # --- Step 3: Standardize & Manual Lasso Prune ---
    start = time.perf_counter()
    print_step(3, 6, f"Standardizing & applying manual Lasso selection (alpha={LASSO_ALPHA})...")

    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train_poly)
    del Z_train_poly

    lasso_idx = manual_lasso_select(Z_train_scaled, y_train, LASSO_MAX_FEATURES)
    Z_train_selected = Z_train_scaled[:, lasso_idx]
    del Z_train_scaled

    print_stat("Features Retained", Z_train_selected.shape[1])
    print_stat("Features Pruned", len(poly_names) - Z_train_selected.shape[1])
    print_time(start)

    # --- Step 4: SGD Tuning (MAE loss) ---
    start = time.perf_counter()
    print_step(4, 6, "Tuning SGDRegressor (epsilon-insensitive) via CV...")

    param_grid = {
        "alpha": SGD_ALPHAS,
        "epsilon": [0.0]   # epsilon=0 => pure MAE loss
    }

    grid_search = GridSearchCV(
        estimator=SGDRegressor(loss='epsilon_insensitive', penalty='l2',
                               max_iter=2000, random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    grid_search.fit(Z_train_selected, y_train)
    model = grid_search.best_estimator_
    best_cv_mae = -grid_search.best_score_

    print_stat("Best Alpha", f"{grid_search.best_params_['alpha']}")
    print_stat("Best CV MAE", f"{best_cv_mae:.4f}")
    print_time(start)

    train_preds = model.predict(Z_train_selected)
    train_nmae = np.sum(np.abs(y_train - train_preds)) / np.sum(np.abs(y_train - np.mean(y_train)))
    train_nmse = np.sum((y_train - train_preds) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    del train_preds, y_train, Z_train_selected

    # --- Step 5: Process Test Data ---
    print_header("PART (C) — TEST PREDICTIONS")
    start = time.perf_counter()
    print_step(5, 6, "Loading and formatting test set...")
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

    # --- Step 6: Execute ---
    start = time.perf_counter()
    print_step(6, 6, "Executing model predictions...")
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

    # --- Final Summary ---
    print_header("FINAL SUMMARY")
    print_stat("Algorithm", "SGDRegressor (NumPy Polynomial Ext. + Manual Lasso)")
    print_stat("Training NMAE", f"{train_nmae:.4f}")
    print_stat("Training NMSE", f"{train_nmse:.4f}")

    if test_nmae is not None:
        print_stat("Test NMAE", f"{test_nmae:.4f}")
        print_stat("Test NMSE", f"{test_nmse:.4f}")

    print_stat("Total Runtime", f"{time.perf_counter() - total_start:.2f}s")
    print("\nNote: Official Public/Private NMAE will be calculated by the grading script.\n" + "=" * 60)

if __name__ == "__main__":
    main()