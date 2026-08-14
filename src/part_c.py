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
LASSO_MAX_ITER = 1500
CV_FOLDS = 3
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
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)

def row_tkeo_mean(x):
    if x.shape[1] < 3: return np.zeros(x.shape[0], dtype=np.float32)
    tkeo = x[:, 1:-1]**2 - (x[:, :-2] * x[:, 2:])
    return np.mean(tkeo, axis=1, dtype=np.float32)

def row_nleo_mean(x):
    if x.shape[1] < 4: return np.zeros(x.shape[0], dtype=np.float32)
    nleo = (x[:, 3:] * x[:, :-3]) - (x[:, 2:-1] * x[:, 1:-2])
    return np.mean(nleo, axis=1, dtype=np.float32)

def row_shannon_entropy(x):
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - mean) / std
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
    window = 8 
    if x.shape[1] < window: return np.zeros(x.shape[0], dtype=np.float32)
    cs = np.cumsum(x, axis=1, dtype=np.float32)
    tonic = (cs[:, window:] - cs[:, :-window]) / window
    phasic = x[:, window:] - tonic
    return np.sum(phasic**2, axis=1).astype(np.float32)

# ============================================================
# 4. Feature Assembly Framework (Base & Polynomial)
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

    # --- 1. BVP Pulse Morphology & SQI ---
    features.append(row_std(bvp)); names.append("bvp_std") # Index 0 used for dead sensor check
    features.append(row_skewness(bvp)); names.append("bvp_skew")
    features.append(row_kurtosis(bvp)); names.append("bvp_kurt")
    features.append(row_zero_crossings(bvp)); names.append("bvp_zcross")
    features.append(row_local_extrema(bvp)); names.append("bvp_extrema")
    features.append(row_shannon_entropy(bvp)); names.append("bvp_entropy_sqi") 

    mad_bvp = np.mean(np.abs(bvp - np.mean(bvp, axis=1, keepdims=True)), axis=1)
    features.append(mad_bvp); names.append("bvp_mad")

    rms_bvp = np.sqrt(np.mean(bvp**2, axis=1)) + 1e-7
    crest_factor = np.max(np.abs(bvp), axis=1) / rms_bvp
    features.append(crest_factor); names.append("bvp_crest_factor")
    
    pulse_pressure = np.percentile(bvp, 95, axis=1) - np.percentile(bvp, 5, axis=1)
    features.append(pulse_pressure); names.append("bvp_pulse_pressure")
    
    L = np.sum(np.abs(vpg), axis=1)
    a = np.max(bvp, axis=1) - np.min(bvp, axis=1)
    features.append(L / (a + 1e-7)); names.append("bvp_katz_fractal")
    features.append(row_nleo_mean(bvp)); names.append("bvp_nleo")

    # --- 2. Hjorth Parameters & Poincare ---
    bvp_std_val = row_std(bvp) + 1e-7
    vpg_std_val = row_std(vpg) + 1e-7
    apg_std_val = row_std(apg) + 1e-7
    
    mobility_bvp = vpg_std_val / bvp_std_val
    mobility_vpg = apg_std_val / vpg_std_val
    complexity_bvp = mobility_vpg / (mobility_bvp + 1e-7)
    
    features.append(mobility_bvp); names.append("bvp_hjorth_mobility")      
    features.append(complexity_bvp); names.append("bvp_hjorth_complexity")  
    
    sd1 = np.std(vpg[:, 1:] - vpg[:, :-1], axis=1) / np.sqrt(2)
    sd2 = np.std(vpg[:, 1:] + vpg[:, :-1], axis=1) / np.sqrt(2)
    features.append(sd1 / (sd2 + 1e-7)); names.append("vpg_poincare_ratio")

    # --- 3. Frequency Equalizer ---
    bpm_targets = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170]
    for bpm in bpm_targets:
        lag = int(60.0 * 64 / bpm)
        features.append(row_autocorr_lag(bvp, lag))
        names.append(f"bvp_ac_{bpm}bpm")

    features.append(row_zero_crossings(vpg)); names.append("vpg_zcross_freq")
    features.append(row_zero_crossings(apg)); names.append("apg_zcross_freq")
    
    ac_energy_low = row_autocorr_lag(bvp, 32) + row_autocorr_lag(bvp, 38)
    ac_energy_high = row_autocorr_lag(bvp, 16) + row_autocorr_lag(bvp, 22)
    features.append(ac_energy_low); names.append("bvp_ac_energy_low")
    features.append(ac_energy_high); names.append("bvp_ac_energy_high")

    # --- 4. VPG & Advanced APG Shape Statistics ---
    features.append(vpg_std_val); names.append("vpg_std")
    features.append(row_skewness(vpg)); names.append("vpg_skew")
    
    features.append(apg_std_val); names.append("apg_std")
    features.append(row_skewness(apg)); names.append("apg_skew")
    features.append(row_kurtosis(apg)); names.append("apg_kurt")
    features.append(row_zero_crossings(apg)); names.append("apg_zcross")
    features.append(row_local_extrema(apg)); names.append("apg_extrema")
    
    est_bpm = (np.sum((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0), axis=1) * 6.0).astype(np.float32)
    features.append(est_bpm); names.append("vpg_est_bpm")

    # --- 5. Context: Motion, Jerk & TKEO ---
    acc_sma_val = row_sma(acc_x, acc_y, acc_z)
    features.append(acc_sma_val); names.append("acc_sma")
    
    acc_sma_ts = np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z)
    features.append(row_std(acc_sma_ts) / (row_mean(acc_sma_ts) + 1e-7)); names.append("acc_dynamic_ratio")
    
    features.append(row_std(acc_x**2 + acc_y**2 + acc_z**2)); names.append("acc_sq_std")
    features.append(row_tkeo_mean(acc_x) + row_tkeo_mean(acc_y) + row_tkeo_mean(acc_z)); names.append("acc_tkeo_total") 
    
    pitch = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2) + 1e-7)
    roll = np.arctan2(acc_x, np.sqrt(acc_y**2 + acc_z**2) + 1e-7)
    features.append(row_mean(pitch)); names.append("acc_pitch_mean")
    features.append(row_mean(roll)); names.append("acc_roll_mean")
    
    jerk_x, jerk_y, jerk_z = np.diff(acc_x, axis=1), np.diff(acc_y, axis=1), np.diff(acc_z, axis=1)
    features.append(row_std(jerk_x) + row_std(jerk_y) + row_std(jerk_z)); names.append("jerk_total_std")
    features.append(row_zero_crossings(acc_x) + row_zero_crossings(acc_y)); names.append("acc_zero_crossings")
    features.append(row_skewness(acc_x)); names.append("acc_x_skew")
    
    features.append(row_std(bvp) * acc_sma_val); names.append("bvp_acc_macro_coupling")

    # --- 6. Stress: Phasic vs Tonic EDA ---
    features.append(row_mean(eda)); names.append("eda_mean")
    features.append(row_std(eda) / (row_mean(eda) + 1e-7)); names.append("eda_baseline_cov")
    features.append(row_std(eda)); names.append("eda_std")
    features.append(row_std(np.diff(eda, axis=1))); names.append("eda_diff_std")
    features.append(row_eda_phasic_energy(eda)); names.append("eda_phasic_energy") 
    features.append(np.sum(np.clip(np.diff(eda, axis=1), 0, None), axis=1)); names.append("eda_pos_scr")

    # --- 7. Temporal Context (Acceleration/Deceleration) ---
    half_bvp = bvp.shape[1] // 2
    half_acc = acc_x.shape[1] // 2
    
    features.append(row_std(bvp[:, :half_bvp])); names.append("bvp_std_h1")
    features.append(row_std(bvp[:, half_bvp:])); names.append("bvp_std_h2")
    features.append(row_sma(acc_x[:, :half_acc], acc_y[:, :half_acc], acc_z[:, :half_acc])); names.append("acc_sma_h1")
    features.append(row_sma(acc_x[:, half_acc:], acc_y[:, half_acc:], acc_z[:, half_acc:])); names.append("acc_sma_h2")

    Z_base = np.column_stack(features).astype(np.float32)
    return Z_base, names

def expand_polynomials(Z_base, names_base):
    poly_features, poly_names = [], []
    n_cols = Z_base.shape[1]

    for i in range(n_cols):
        poly_features.append(Z_base[:, i])
        poly_names.append(names_base[i])

    for i in range(n_cols):
        for j in range(i, n_cols):
            poly_features.append(Z_base[:, i] * Z_base[:, j])
            if i == j:
                poly_names.append(f"{names_base[i]}^2")
            else:
                poly_names.append(f"{names_base[i]}*{names_base[j]}")

    Z_poly = np.column_stack(poly_features).astype(np.float32)
    if not np.all(np.isfinite(Z_poly)): raise ValueError("Poly matrix contains NaN/Inf.")
    return Z_poly, poly_names

# ============================================================
# 5. Main Pipeline Execution
# ============================================================
def main():
    if len(sys.argv) != 4: sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")
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
    
    # Pre-calculate global mean for dead sensor fallback
    GLOBAL_MEAN_HR = np.mean(y_train)
    
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

    # --- Step 3: Standardize & Lasso Prune ---
    start = time.perf_counter()
    print_step(3, 6, f"Standardizing & applying Lasso Selection (alpha={LASSO_ALPHA})...")
    
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train_poly)
    del Z_train_poly
    
    selector = SelectFromModel(
        Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER, random_state=RANDOM_STATE, tol=0.001), 
        prefit=False
    )
    Z_train_selected = selector.fit_transform(Z_train_scaled, y_train)
    del Z_train_scaled
    
    print_stat("Features Retained", Z_train_selected.shape[1])
    print_stat("Features Pruned", len(poly_names) - Z_train_selected.shape[1])
    print_time(start)

    # --- Step 4: SGD Tuning ---
    start = time.perf_counter()
    print_step(4, 6, f"Tuning SGDRegressor (epsilon-insensitive) via {CV_FOLDS}-Fold CV...")

    param_grid = {
        "alpha": [1e-4, 1e-3], 
        "epsilon": [0.01, 0.05, 0.1],
        "l1_ratio": [0.15, 0.5]
    }

    grid_search = GridSearchCV(
        estimator=SGDRegressor(loss='epsilon_insensitive', penalty='elasticnet', max_iter=2000, random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error", 
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE), 
        n_jobs=1
    )
    grid_search.fit(Z_train_selected, y_train)
    model, best_cv_mae = grid_search.best_estimator_, -grid_search.best_score_
    
    print_stat("Best Params", f"alpha={grid_search.best_params_['alpha']}, epsilon={grid_search.best_params_['epsilon']}, l1_ratio={grid_search.best_params_['l1_ratio']}")
    print_stat("Best CV MAE", f"{best_cv_mae:.4f}")
    print_time(start)

    train_preds = model.predict(Z_train_selected)
    train_nmae = np.sum(np.abs(y_train - train_preds)) / np.sum(np.abs(y_train - np.mean(y_train)))
    train_nmse = np.sum((y_train - train_preds) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    del train_preds, Z_train_selected

    # --- Step 5: Process Test Data ---
    print_header("PART (C) — TEST PREDICTIONS")
    start = time.perf_counter()
    print_step(5, 6, "Loading and formatting test set...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    Z_test_base, _ = extract_base_features(X_test_raw, feature_columns)
    del X_test_raw
    
    # Dead Sensor identification (bvp_std is index 0)
    bvp_std_test = Z_test_base[:, 0]
    dead_sensor_mask = (bvp_std_test < 1e-5)
    
    Z_test_poly, _ = expand_polynomials(Z_test_base, base_names)
    del Z_test_base

    Z_test_selected = selector.transform(scaler.transform(Z_test_poly))
    del Z_test_poly
    
    print_stat("Test Samples", f"{Z_test_selected.shape[0]:,}")
    print_time(start)

    # --- Step 6: Execute ---
    start = time.perf_counter()
    print_step(6, 6, "Executing model predictions...")
    
    predictions = model.predict(Z_test_selected)
    
    num_dead_sensors = np.sum(dead_sensor_mask)
    if num_dead_sensors > 0:
        predictions[dead_sensor_mask] = GLOBAL_MEAN_HR
        print_stat("Dead Sensors Handled", num_dead_sensors)
    
    if not np.all(np.isfinite(predictions)): raise ValueError("Predictions contain NaN/Inf.")
    
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print_stat("Predictions Saved To", predictions_path)
    print_time(start)

    # --- Final Summary ---
    print_header("FINAL SUMMARY")
    print_stat("Algorithm", "SGDRegressor (NumPy Polynomial Ext. + Lasso)")
    print_stat("Training NMAE", f"{train_nmae:.4f}")
    print_stat("Training NMSE", f"{train_nmse:.4f}")
    print_stat("Total Runtime", f"{time.perf_counter() - total_start:.2f}s")
    print("\nNote: Official Public/Private NMAE will be calculated by the grading script.\n" + "=" * 60)

if __name__ == "__main__":
    main()