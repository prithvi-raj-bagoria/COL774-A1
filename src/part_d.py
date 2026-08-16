%%writefile src/part_d.py
import sys
import time
import pickle
import warnings
import gc
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Configuration
# ============================================================
LASSO_MAX_FEATURES = 495
SUBSAMPLE_SIZE = 100_000
RANDOM_STATE = 42

LASSO_ALPHA = 0.001
LASSO_MAX_ITER = 3000
LASSO_TOL = 1e-4

CV_FOLDS = 3
SGD_ALPHAS = [1e-5, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]

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
# Ultra-Fast 1D Math Helpers (Vectorized over Axis 1)
# ============================================================
def fast_skew(x, m, s):
    return np.mean(((x - m[:, None]) / (s[:, None] + 1e-7))**3, axis=1)

def fast_kurt(x, m, s):
    return np.mean(((x - m[:, None]) / (s[:, None] + 1e-7))**4, axis=1)

def fast_zcross(x, m):
    c = x - m[:, None]
    return np.sum((c[:, :-1] * c[:, 1:]) < 0, axis=1)

def fast_tkeo_mean(x):
    if x.shape[1] < 3: return np.zeros(x.shape[0], dtype=np.float32)
    return np.mean(x[:, 1:-1]**2 - (x[:, :-2] * x[:, 2:]), axis=1)

# ============================================================
# Feature Extraction (Modality-by-Modality & Dimension-Safe)
# ============================================================
def extract_features(filepath):
    features, names = [], []

    def add_feat(val_array, name):
        # np.ravel() guarantees the feature is strictly a 1D array (N,)
        # preventing any np.column_stack dimension crashing!
        features.append(np.ravel(val_array))
        names.append(name)
        
    def get_stats(x):
        return np.mean(x, axis=1), np.std(x, axis=1), np.min(x, axis=1), np.max(x, axis=1)

    with np.load(filepath, allow_pickle=False) as data:
        y = data["glucose"] if "glucose" in data.files else None
        
        # ----------------------------------------------------
        # 1. Skin Temperature (e4_temp)
        # .reshape(N, -1) destroys any empty trailing dimensions
        temp = data["e4_temp"].reshape(data["e4_temp"].shape[0], -1)
        m_temp, s_temp, min_t, max_t = get_stats(temp)
        add_feat(m_temp, "temp_mean")
        add_feat(np.median(temp, axis=1), "temp_median")
        add_feat(s_temp, "temp_std")
        add_feat(min_t, "temp_min")
        add_feat(max_t, "temp_max")
        add_feat(np.mean(temp[:, :240], axis=1), "temp_start_mean")
        add_feat(np.mean(temp[:, -240:], axis=1), "temp_end_mean")
        add_feat(np.mean(temp[:, -240:], axis=1) - np.mean(temp[:, :240], axis=1), "temp_slope")
        pct_t = np.percentile(temp, [25, 75], axis=1)
        add_feat(pct_t[1] - pct_t[0], "temp_iqr")
        add_feat(np.sum(np.abs(np.diff(temp, axis=1)), axis=1), "temp_line_len")
        del temp; gc.collect()

        # ----------------------------------------------------
        # 2. Electrodermal Activity (e4_eda)
        eda = data["e4_eda"].reshape(data["e4_eda"].shape[0], -1)
        m_eda, s_eda, min_e, max_e = get_stats(eda)
        add_feat(m_eda, "eda_mean")
        add_feat(s_eda, "eda_std")
        add_feat(max_e, "eda_max")
        add_feat(min_e, "eda_min")
        pct_e = np.percentile(eda, [25, 75], axis=1)
        add_feat(pct_e[1] - pct_e[0], "eda_iqr")
        add_feat(np.mean(eda[:, -240:], axis=1) - np.mean(eda[:, :240], axis=1), "eda_slope")
        diff_eda = np.diff(eda, axis=1)
        add_feat(np.sum(np.maximum(0, diff_eda), axis=1), "eda_pos_deriv")
        add_feat(max_e - m_eda, "eda_phasic_max")
        add_feat(np.sum(np.abs(diff_eda), axis=1), "eda_line_len")
        add_feat(fast_skew(eda, m_eda, s_eda), "eda_skew")
        del eda, diff_eda; gc.collect()

        # ----------------------------------------------------
        # 3. Heart Rate (e4_hr)
        hr = data["e4_hr"].reshape(data["e4_hr"].shape[0], -1)
        m_hr, s_hr, min_h, max_h = get_stats(hr)
        add_feat(m_hr, "hr_mean")
        add_feat(s_hr, "hr_std")
        add_feat(max_h, "hr_max")
        add_feat(min_h, "hr_min")
        pct_h = np.percentile(hr, [25, 75], axis=1)
        add_feat(pct_h[1] - pct_h[0], "hr_iqr")
        add_feat(np.mean(hr[:, -60:], axis=1) - np.mean(hr[:, :60], axis=1), "hr_slope")
        add_feat(fast_zcross(hr, m_hr), "hr_zcross")
        add_feat(fast_skew(hr, m_hr, s_hr), "hr_skew")
        add_feat(fast_kurt(hr, m_hr, s_hr), "hr_kurt")
        add_feat(np.sqrt(np.mean(hr**2, axis=1)), "hr_rms")
        del hr; gc.collect()

        # ----------------------------------------------------
        # 4. Blood Volume Pulse (e4_bvp)
        bvp = data["e4_bvp"].reshape(data["e4_bvp"].shape[0], -1)
        m_bvp, s_bvp, _, _ = get_stats(bvp)
        diff_bvp = np.diff(bvp, axis=1)
        diff2_bvp = np.diff(diff_bvp, axis=1)
        
        add_feat(s_bvp, "bvp_ac_amp")
        add_feat(fast_skew(bvp, m_bvp, s_bvp), "bvp_skew")
        add_feat(fast_kurt(bvp, m_bvp, s_bvp), "bvp_kurt")
        add_feat(np.mean(np.abs(bvp), axis=1), "bvp_tonic_abs")
        add_feat(np.sum(np.abs(diff_bvp), axis=1), "bvp_line_len")
        add_feat(np.max(np.abs(bvp), axis=1), "bvp_max_abs")
        pct_bvp = np.percentile(bvp, [10, 90], axis=1)
        add_feat(pct_bvp[1] - pct_bvp[0], "bvp_p90_p10")
        add_feat(fast_zcross(bvp, m_bvp), "bvp_zcross")
        add_feat(np.var(diff_bvp, axis=1), "bvp_vpg_var")
        add_feat(np.var(diff2_bvp, axis=1), "bvp_apg_var")
        del bvp, diff_bvp, diff2_bvp; gc.collect()

        # ----------------------------------------------------
        # 5. E4 Accelerometer Magnitude (e4_acc)
        # Force strict 3D format: (N, 9600, 3)
        acc_raw = data["e4_acc"].reshape(data["e4_acc"].shape[0], -1, 3)
        acc_mag = np.sqrt(acc_raw[:, :, 0]**2 + acc_raw[:, :, 1]**2 + acc_raw[:, :, 2]**2)
        del acc_raw; gc.collect()
        
        m_acc, s_acc, min_a, max_a = get_stats(acc_mag)
        add_feat(m_acc, "acc_mean")
        add_feat(s_acc, "acc_std")
        add_feat(np.var(acc_mag, axis=1), "acc_var")
        add_feat(max_a, "acc_max")
        add_feat(min_a, "acc_min")
        pct_a = np.percentile(acc_mag, [25, 75, 90], axis=1)
        add_feat(pct_a[1] - pct_a[0], "acc_iqr")
        add_feat(fast_skew(acc_mag, m_acc, s_acc), "acc_skew")
        add_feat(np.mean(np.abs(np.diff(acc_mag, axis=1)), axis=1), "acc_mean_jerk")
        add_feat(fast_zcross(acc_mag, m_acc), "acc_zcross")
        add_feat(pct_a[2], "acc_p90")
        del acc_mag; gc.collect()

        # ----------------------------------------------------
        # 6. Zephyr ECG (zephyr_ecg)
        ecg = data["zephyr_ecg"].reshape(data["zephyr_ecg"].shape[0], -1)
        m_ecg, s_ecg, min_ecg, max_ecg = get_stats(ecg)
        add_feat(m_ecg, "ecg_mean")
        add_feat(s_ecg, "ecg_std")
        add_feat(max_ecg, "ecg_max")
        add_feat(min_ecg, "ecg_min")
        pct_ecg = np.percentile(ecg, [25, 75], axis=1)
        add_feat(pct_ecg[1] - pct_ecg[0], "ecg_iqr")
        add_feat(fast_skew(ecg, m_ecg, s_ecg), "ecg_skew")
        add_feat(fast_kurt(ecg, m_ecg, s_ecg), "ecg_kurt")
        add_feat(fast_zcross(ecg, m_ecg), "ecg_zcross")
        add_feat(np.var(np.diff(ecg, axis=1), axis=1), "ecg_diff_var")
        add_feat(fast_tkeo_mean(ecg), "ecg_tkeo_mean")
        del ecg; gc.collect()

        # ----------------------------------------------------
        # 7. Zephyr Respiration (zephyr_breathing)
        breath = data["zephyr_breathing"].reshape(data["zephyr_breathing"].shape[0], -1)
        m_br, s_br, min_br, max_br = get_stats(breath)
        add_feat(m_br, "breath_mean")
        add_feat(s_br, "breath_std")
        add_feat(max_br, "breath_max")
        add_feat(min_br, "breath_min")
        pct_br = np.percentile(breath, [25, 75], axis=1)
        add_feat(pct_br[1] - pct_br[0], "breath_iqr")
        add_feat(np.mean(breath[:, -1500:], axis=1) - np.mean(breath[:, :1500], axis=1), "breath_slope")
        add_feat(fast_skew(breath, m_br, s_br), "breath_skew")
        add_feat(fast_kurt(breath, m_br, s_br), "breath_kurt")
        add_feat(fast_zcross(breath, m_br), "breath_zcross")
        add_feat(np.mean(np.abs(np.diff(breath, axis=1)), axis=1), "breath_mean_jerk")
        del breath; gc.collect()

        # ----------------------------------------------------
        # 8. Zephyr Chest Accelerometer Magnitude (zephyr_acc)
        zacc_raw = data["zephyr_acc"].reshape(data["zephyr_acc"].shape[0], -1, 3)
        zacc_mag = np.sqrt(zacc_raw[:, :, 0]**2 + zacc_raw[:, :, 1]**2 + zacc_raw[:, :, 2]**2)
        del zacc_raw; gc.collect()
        
        m_zacc, s_zacc, min_za, max_za = get_stats(zacc_mag)
        add_feat(m_zacc, "zacc_mean")
        add_feat(s_zacc, "zacc_std")
        add_feat(np.var(zacc_mag, axis=1), "zacc_var")
        add_feat(max_za, "zacc_max")
        add_feat(min_za, "zacc_min")
        pct_za = np.percentile(zacc_mag, [25, 75, 90], axis=1)
        add_feat(pct_za[1] - pct_za[0], "zacc_iqr")
        add_feat(fast_skew(zacc_mag, m_zacc, s_zacc), "zacc_skew")
        add_feat(np.mean(np.abs(np.diff(zacc_mag, axis=1)), axis=1), "zacc_mean_jerk")
        add_feat(fast_zcross(zacc_mag, m_zacc), "zacc_zcross")
        add_feat(pct_za[2], "zacc_p90")
        del zacc_mag; gc.collect()

    # All arrays safely cleared from RAM, purely 1D features remaining!
    X_file = np.column_stack(features).astype(np.float32)
    return X_file, y, names

# ============================================================
# Polynomial Expansion
# ============================================================
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
            
    return out, out_names

# ============================================================
# Training Pipeline
# ============================================================
def run_training(protocol, input_dir, output_model_path):
    total_start = time.perf_counter()
    print_header(f"PART (D) — TRAINING PROTOCOL: {protocol.upper()}")

    start = time.perf_counter()
    print_step(1, 5, "Extracting Base Features from .npz files...")
    all_X, all_y = [], []
    base_names = None
    
    files = sorted(Path(input_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {input_dir}")
        
    for fpath in files:
        X_file, y_file, nms = extract_features(fpath)
        all_X.append(X_file)
        if y_file is not None:
            all_y.append(y_file)
        if base_names is None:
            base_names = nms

    Z_raw = np.vstack(all_X)
    Y_target = np.concatenate(all_y) if all_y else None
    
    print_stat("Total Samples", Z_raw.shape[0])
    print_stat("Base Features", Z_raw.shape[1])
    print_time(start)

    start = time.perf_counter()
    print_step(2, 5, "Scaling Base Features...")
    scaler_base = StandardScaler()
    Z_scaled_base = scaler_base.fit_transform(Z_raw)
    print_time(start)

    start = time.perf_counter()
    print_step(3, 5, "Expanding Degree-2 Polynomial Interactions...")
    Z_poly, poly_names = expand_polynomials(Z_scaled_base, base_names)
    scaler_poly = StandardScaler()
    Z_poly_scaled = scaler_poly.fit_transform(Z_poly)
    print_stat("Expanded Features", Z_poly_scaled.shape[1])
    print_time(start)

    start = time.perf_counter()
    print_step(4, 5, f"Lasso Selection (Shrinking to {LASSO_MAX_FEATURES} features)...")
    np.random.seed(RANDOM_STATE)
    if Z_poly_scaled.shape[0] > SUBSAMPLE_SIZE:
        idx = np.random.choice(Z_poly_scaled.shape[0], SUBSAMPLE_SIZE, replace=False)
        X_sub, y_sub = Z_poly_scaled[idx], Y_target[idx]
    else:
        X_sub, y_sub = Z_poly_scaled, Y_target

    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL, random_state=RANDOM_STATE)
    lasso.fit(X_sub, y_sub)

    coef_abs = np.abs(lasso.coef_)
    top_indices = np.argsort(coef_abs)[-LASSO_MAX_FEATURES:]
    top_indices = np.sort(top_indices)
    
    Z_final_train = Z_poly_scaled[:, top_indices]
    final_feature_names = [poly_names[i] for i in top_indices]
    print_stat("Final Feature Count", Z_final_train.shape[1])
    print_time(start)

    start = time.perf_counter()
    print_step(5, 5, "Tuning Linear SGDRegressor via CV...")
    grid = GridSearchCV(
        estimator=SGDRegressor(loss='epsilon_insensitive', epsilon=0.0, penalty='l2', random_state=RANDOM_STATE),
        param_grid={"alpha": SGD_ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    grid.fit(Z_final_train, Y_target)
    model = grid.best_estimator_
    
    print_stat("Best Alpha", grid.best_params_['alpha'])
    print_stat("CV MAE", f"{-grid.best_score_:.4f}")

    state_dict = {
        "format_version": 1,
        "protocol": protocol,
        "intercept": float(model.intercept_[0] if isinstance(model.intercept_, np.ndarray) else model.intercept_),
        "coef": np.array(model.coef_, dtype=np.float64),
        "feature_names": final_feature_names,
        "preprocessing_state": {
            "scaler_base": scaler_base,
            "base_names": base_names,
            "scaler_poly": scaler_poly,
            "top_indices": top_indices
        }
    }
    with open(output_model_path, "wb") as f:
        pickle.dump(state_dict, f)
    
    print_stat("Model Saved To", output_model_path)
    print_time(start)
    print_header(f"TRAINING COMPLETE ({time.perf_counter() - total_start:.2f}s total)")

# ============================================================
# Feature Engineering Pipeline
# ============================================================
def run_feature_engineering(protocol, input_dir, model_path, output_npy_path):
    total_start = time.perf_counter()
    print_header(f"PART (D) — FEATURE ENGINEERING: {protocol.upper()}")

    print_step(1, 3, "Loading saved state...")
    with open(model_path, "rb") as f:
        state_dict = pickle.load(f)
    
    if state_dict["protocol"] != protocol:
        raise ValueError(f"Protocol mismatch: Expected {protocol}, found {state_dict['protocol']} in pickle.")

    prep = state_dict["preprocessing_state"]
    scaler_base = prep["scaler_base"]
    base_names = prep["base_names"]
    scaler_poly = prep["scaler_poly"]
    top_indices = prep["top_indices"]

    print_step(2, 3, "Processing test files sequentially...")
    all_X = []
    files = sorted(Path(input_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {input_dir}")
        
    for fpath in files:
        X_file, _, _ = extract_features(fpath)
        all_X.append(X_file)
        
    Z_raw = np.vstack(all_X)
    print_stat("Test Samples", Z_raw.shape[0])

    print_step(3, 3, "Applying transformations...")
    Z_test = scaler_base.transform(Z_raw)
    Z_poly_test, _ = expand_polynomials(Z_test, base_names)
    Z_poly_test = scaler_poly.transform(Z_poly_test)
    Z_final_test = Z_poly_test[:, top_indices]

    np.save(output_npy_path, Z_final_test)
    print_stat("Features Saved To", output_npy_path)
    print_stat("Final Shape", Z_final_test.shape)
    print_header(f"TESTING COMPLETE ({time.perf_counter() - total_start:.2f}s total)")

# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit("Usage:\n"
                 "Train: python3 part_d.py train <protocol> <input_dir> <model.pkl>\n"
                 "Test:  python3 part_d.py feature_engineering <protocol> <input_dir> <model.pkl> <out.npy>")

    mode = sys.argv[1].lower()
    protocol = sys.argv[2].lower()
    input_dir = sys.argv[3]
    model_path = sys.argv[4]

    if mode == "train":
        run_training(protocol, input_dir, model_path)
    elif mode == "feature_engineering":
        if len(sys.argv) != 6:
            sys.exit("Test mode requires 6 arguments including the output .npy path.")
        output_npy = sys.argv[5]
        run_feature_engineering(protocol, input_dir, model_path, output_npy)
    else:
        sys.exit(f"Unknown mode: {mode}. Must be 'train' or 'feature_engineering'.")