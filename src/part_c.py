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

def row_sma(x, y, z):
    """Signal Magnitude Area: Actigraphy standard for physical exertion."""
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)

# ============================================================
# 4. Feature Assembly Framework (Base 20 & Cubic Expansion)
# ============================================================
def extract_base_features(X_raw, feature_columns):
    """Extracts exactly 20 elite biological features to prepare for k=3 expansion."""
    features, names = [], []
    
    acc_x = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]]
    acc_y = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]]
    acc_z = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]]
    bvp = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]]
    eda = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]]

    # 1. BVP Pulse Morphology (3)
    features.append(row_std(bvp)); names.append("bvp_std")
    features.append(row_skewness(bvp)); names.append("bvp_skew")
    features.append(row_kurtosis(bvp)); names.append("bvp_kurt")

    # 2. VPG Velocity (2)
    vpg = np.diff(bvp, axis=1)
    features.append(row_std(vpg)); names.append("vpg_std")
    
    est_bpm = (np.sum((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0), axis=1) * 6.0).astype(np.float32)
    features.append(est_bpm); names.append("vpg_est_bpm")

    # 3. Frequency Equalizer (7)
    bpm_targets = [60, 80, 100, 120, 140, 160, 180]
    for bpm in bpm_targets:
        lag = int(60.0 * 64 / bpm)
        features.append(row_autocorr_lag(bvp, lag))
        names.append(f"bvp_ac_{bpm}bpm")

    # 4. Context: Motion & Stress (4)
    acc_sma_val = row_sma(acc_x, acc_y, acc_z)
    features.append(acc_sma_val); names.append("acc_sma")
    
    acc_sq = acc_x**2 + acc_y**2 + acc_z**2
    features.append(row_std(acc_sq)); names.append("acc_sq_std")
    
    features.append(row_mean(eda)); names.append("eda_mean")
    features.append(row_std(eda)); names.append("eda_std")

    # 5. Temporal Context (4)
    half_bvp = bvp.shape[1] // 2
    half_acc = acc_x.shape[1] // 2
    
    features.append(row_std(bvp[:, :half_bvp])); names.append("bvp_std_h1")
    features.append(row_std(bvp[:, half_bvp:])); names.append("bvp_std_h2")
    features.append(row_sma(acc_x[:, :half_acc], acc_y[:, :half_acc], acc_z[:, :half_acc])); names.append("acc_sma_h1")
    features.append(row_sma(acc_x[:, half_acc:], acc_y[:, half_acc:], acc_z[:, half_acc:])); names.append("acc_sma_h2")

    Z_base = np.column_stack(features).astype(np.float32)
    return Z_base, names

def expand_polynomials_cubic(Z_base, names_base):
    """
    100% Legal Pure-NumPy Cubic Expander (k=3).
    Generates Degrees 1, 2, and 3 manually (x, x^2, x*y, x^3, x^2*y, x*y*z).
    20 base features -> exactly 1770 expanded features.
    """
    poly_features, poly_names = [], []
    n_cols = Z_base.shape[1]

    # 1. Degree 1
    for i in range(n_cols):
        poly_features.append(Z_base[:, i])
        poly_names.append(names_base[i])

    # 2. Degree 2
    for i in range(n_cols):
        for j in range(i, n_cols):
            poly_features.append(Z_base[:, i] * Z_base[:, j])
            if i == j:
                poly_names.append(f"{names_base[i]}^2")
            else:
                poly_names.append(f"{names_base[i]}*{names_base[j]}")
                
    # 3. Degree 3
    for i in range(n_cols):
        for j in range(i, n_cols):
            for k in range(j, n_cols):
                poly_features.append(Z_base[:, i] * Z_base[:, j] * Z_base[:, k])
                if i == j == k:
                    poly_names.append(f"{names_base[i]}^3")
                elif i == j:
                    poly_names.append(f"{names_base[i]}^2*{names_base[k]}")
                elif j == k:
                    poly_names.append(f"{names_base[i]}*{names_base[j]}^2")
                else:
                    poly_names.append(f"{names_base[i]}*{names_base[j]}*{names_base[k]}")

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

    # --- Step 2: Extract & Expand ---
    start = time.perf_counter()
    print_step(2, 6, "Building Base 20 & Cubic Feature Matrix (k=3)...")
    
    Z_train_base, base_names = extract_base_features(X_train_raw, feature_columns)
    del X_train_raw
    
    Z_train_poly, poly_names = expand_polynomials_cubic(Z_train_base, base_names)
    del Z_train_base
    
    print_stat("Base Features", len(base_names))
    print_stat("Expanded Cubic Features", Z_train_poly.shape[1])
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
    print_step(4, 6, "Tuning SGDRegressor (epsilon-insensitive) via 5-Fold CV...")

    param_grid = {
        "alpha": [1e-4, 1e-3], 
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

    # --- Step 5: Process Test Data ---
    print_header("PART (C) — TEST PREDICTIONS")
    start = time.perf_counter()
    print_step(5, 6, "Loading and formatting test set...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    Z_test_base, _ = extract_base_features(X_test_raw, feature_columns)
    del X_test_raw
    Z_test_poly, _ = expand_polynomials_cubic(Z_test_base, base_names)
    del Z_test_base

    Z_test_selected = selector.transform(scaler.transform(Z_test_poly))
    del Z_test_poly
    
    print_stat("Test Samples", f"{Z_test_selected.shape[0]:,}")
    print_time(start)

    # --- Step 6: Execute ---
    start = time.perf_counter()
    print_step(6, 6, "Executing model predictions...")
    predictions = model.predict(Z_test_selected)
    if not np.all(np.isfinite(predictions)): raise ValueError("Predictions contain NaN/Inf.")
    
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print_stat("Predictions Saved To", predictions_path)
    print_time(start)

    # --- Final Summary ---
    print_header("FINAL SUMMARY")
    print_stat("Algorithm", "SGDRegressor (Cubic Polynomial Ext. + Lasso)")
    print_stat("Training NMAE", f"{train_nmae:.4f}")
    print_stat("Training NMSE", f"{train_nmse:.4f}")
    print_stat("Total Runtime", f"{time.perf_counter() - total_start:.2f}s")
    print("\nNote: Official Public/Private NMAE will be calculated by the grading script.\n" + "=" * 60)

if __name__ == "__main__":
    main()