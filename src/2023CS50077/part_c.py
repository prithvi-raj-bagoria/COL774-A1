import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

EXPECTED_RAW_FEATURES = 1640
LASSO_MAX_FEATURES = 500
RANDOM_STATE = 42
LASSO_ALPHA = 0.001
LASSO_MAX_ITER = 3000
LASSO_TOL = 1e-4
SUBSAMPLE_SIZE = 100_000
CV_FOLDS = 3
SGD_ALPHAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-2, 1.0]

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
    if x.shape[1] <= lag: return np.zeros(x.shape[0], dtype=np.float32)
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return (np.sum(c[:, :-lag] * c[:, lag:], axis=1) / (np.sum(c * c, axis=1) + 1e-10)).astype(np.float32)
def row_zero_crossings(x):
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum((c[:, :-1] * c[:, 1:]) < 0, axis=1).astype(np.float32)
def row_local_extrema(x):
    d = np.diff(x, axis=1)
    return np.sum((d[:, :-1] * d[:, 1:]) < 0, axis=1).astype(np.float32)
def row_sma(x, y, z): return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)
def row_tkeo_mean(x):
    if x.shape[1] < 3: return np.zeros(x.shape[0], dtype=np.float32)
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
def row_rms(x): return np.sqrt(np.mean(x**2, axis=1, dtype=np.float32))
def row_variance(x): return np.var(x, axis=1, dtype=np.float32)
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
    var_x = np.var(ax, axis=1); var_y = np.var(ay, axis=1); var_z = np.var(az, axis=1)
    return np.maximum.reduce([var_x, var_y, var_z]) / (var_x + var_y + var_z + 1e-10)
def row_postural_transitions(acc_x, acc_y, acc_z):
    ax_abs = np.abs(acc_x); ay_abs = np.abs(acc_y); az_abs = np.abs(acc_z)
    dominant = np.argmax(np.stack([ax_abs, ay_abs, az_abs], axis=2), axis=2)
    return np.sum(dominant[:, :-1] != dominant[:, 1:], axis=1).astype(np.float32)
def row_percentile(x, q): return np.percentile(x, q, axis=1).astype(np.float32)
def row_turning_point_ratio(x): return row_local_extrema(x) / x.shape[1]

def extract_base_features(X_raw, feature_columns):
    features, names = [], []
    acc_x = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]]
    acc_y = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]]
    acc_z = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]]
    bvp = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]]
    eda = X_raw[:, [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]]

    vpg = np.diff(bvp, axis=1); apg = np.diff(vpg, axis=1)
    acc_sq = acc_x**2 + acc_y**2 + acc_z**2
    acc_mag = np.sqrt(acc_sq)

    def add_feat(val, name):
        features.append(np.asarray(val, dtype=np.float32)); names.append(name)

    add_feat(row_skewness(bvp), "bvp_skew")
    add_feat(row_kurtosis(bvp), "bvp_kurt")
    add_feat(row_mad(bvp), "bvp_mad")
    add_feat(row_percentile(bvp, 75) - row_percentile(bvp, 25), "bvp_iqr")
    add_feat(np.max(bvp, axis=1) - np.min(bvp, axis=1), "bvp_peak2peak")
    add_feat(row_percentile(bvp, 90) - row_percentile(bvp, 10), "bvp_p90_p10")
    add_feat(np.max(np.abs(bvp), axis=1) / (row_rms(bvp) + 1e-7), "bvp_crest_factor")
    add_feat(row_rms(bvp), "bvp_rms")
    add_feat(row_mean(bvp), "bvp_mean")

    add_feat(row_mean(np.abs(vpg)), "vpg_mean_abs")
    add_feat(row_variance(vpg), "vpg_var")
    add_feat(row_skewness(vpg), "vpg_skew")
    add_feat(row_kurtosis(vpg), "vpg_kurt")
    add_feat(np.max(vpg, axis=1), "vpg_max")
    add_feat(np.min(vpg, axis=1), "vpg_min")
    add_feat(row_zero_crossings(vpg), "vpg_zcross")
    add_feat(np.sum(np.abs(np.diff(vpg, axis=1)), axis=1), "vpg_line_length")

    add_feat(row_variance(apg), "apg_var")
    add_feat(row_mean(np.abs(apg)), "apg_mean_abs")
    add_feat(row_skewness(apg), "apg_skew")
    add_feat(row_kurtosis(apg), "apg_kurt")
    add_feat(row_zero_crossings(apg), "apg_zcross")
    add_feat(row_turning_point_ratio(apg), "apg_turning_ratio")

    ac32 = row_autocorr_lag(bvp, 32); ac38 = row_autocorr_lag(bvp, 38); ac48 = row_autocorr_lag(bvp, 48); ac64 = row_autocorr_lag(bvp, 64); ac77 = row_autocorr_lag(bvp, 77)
    add_feat(ac32, "bvp_ac_32")
    add_feat(ac38, "bvp_ac_38")
    add_feat(ac48, "bvp_ac_48")
    add_feat(ac64, "bvp_ac_64")
    add_feat(ac77, "bvp_ac_77")
    add_feat((ac32 + ac38) / (ac64 + ac77 + 1e-7), "bvp_ac_high_low_ratio")

    add_feat(row_shannon_entropy(bvp), "bvp_entropy")
    add_feat(row_tkeo_mean(bvp), "bvp_tkeo")
    bvp_std = row_std(bvp) + 1e-7
    vpg_std = row_std(vpg) + 1e-7
    apg_std = row_std(apg) + 1e-7
    hj_mob = vpg_std / bvp_std
    add_feat(hj_mob, "bvp_hjorth_mob")
    add_feat((apg_std / vpg_std) / hj_mob, "bvp_hjorth_com")

    add_feat(row_sma(acc_x, acc_y, acc_z), "acc_sma")
    add_feat(row_mad(acc_mag), "acc_mad")
    add_feat(row_variance(acc_sq), "acc_norm_sq_var")
    add_feat(row_rms(acc_mag), "acc_rms")
    add_feat(row_tkeo_mean(acc_x) + row_tkeo_mean(acc_y) + row_tkeo_mean(acc_z), "acc_tkeo")
    jerk_mag = np.sqrt(np.diff(acc_x, axis=1)**2 + np.diff(acc_y, axis=1)**2 + np.diff(acc_z, axis=1)**2)
    add_feat(row_mean(jerk_mag), "acc_jerk_mean_abs")
    snap_mag = np.sqrt(np.diff(acc_x, n=2, axis=1)**2 + np.diff(acc_y, n=2, axis=1)**2 + np.diff(acc_z, n=2, axis=1)**2)
    add_feat(row_mean(snap_mag), "acc_snap_mean_abs")
    add_feat(row_std(acc_mag) / (row_mean(acc_mag) + 1e-7), "acc_cv")
    add_feat(np.sum(np.abs(acc_mag - row_mean(acc_mag).reshape(-1, 1)), axis=1).astype(np.float32), "acc_vib_int")

    add_feat(row_mean(acc_x), "acc_mean_x")
    add_feat(row_mean(acc_y), "acc_mean_y")
    add_feat(row_mean(acc_z), "acc_mean_z")
    roll, pitch = row_roll_pitch(acc_x, acc_y, acc_z)
    add_feat(row_mean(roll), "acc_roll")
    add_feat(row_mean(pitch), "acc_pitch")
    add_feat(row_cross_corr(acc_x, acc_y), "acc_corr_xy")
    add_feat(row_cross_corr(acc_x, acc_z), "acc_corr_xz")
    add_feat(row_cross_corr(acc_y, acc_z), "acc_corr_yz")
    add_feat(row_dominant_axis_ratio(acc_x, acc_y, acc_z), "acc_dom_axis_ratio")
    add_feat(row_postural_transitions(acc_x, acc_y, acc_z), "acc_post_transitions")

    add_feat(row_skewness(acc_mag), "acc_skew")
    add_feat(row_kurtosis(acc_mag), "acc_kurt")
    add_feat(row_percentile(acc_mag, 75) - row_percentile(acc_mag, 25), "acc_iqr")
    add_feat(np.max(acc_mag, axis=1) / (row_rms(acc_mag) + 1e-7), "acc_crest_factor")
    add_feat(np.max(acc_mag, axis=1) - np.min(acc_mag, axis=1), "acc_peak2peak")
    add_feat(row_percentile(acc_mag, 10), "acc_p10")
    add_feat(row_percentile(acc_mag, 90), "acc_p90")

    add_feat(row_zero_crossings(acc_mag), "acc_zcross")
    add_feat(row_turning_point_ratio(acc_mag), "acc_turning_ratio")
    diff_mag = np.diff(acc_mag, axis=1); diff2_mag = np.diff(diff_mag, axis=1)
    hj_mob_acc = row_std(diff_mag) / (row_std(acc_mag) + 1e-7)
    add_feat(hj_mob_acc, "acc_hjorth_mob")
    add_feat((row_std(diff2_mag) / (row_std(diff_mag) + 1e-7)) / (hj_mob_acc + 1e-7), "acc_hjorth_com")
    add_feat(row_autocorr_lag(acc_mag, 16), "acc_ac_16")
    add_feat(row_autocorr_lag(acc_mag, 32), "acc_ac_32")
    add_feat(row_shannon_entropy(acc_mag), "acc_entropy")

    add_feat(row_mean(eda), "eda_mean")
    add_feat(row_std(eda), "eda_std")
    add_feat(np.min(eda, axis=1), "eda_min")
    add_feat(np.max(eda, axis=1), "eda_max")
    add_feat(row_percentile(eda, 75) - row_percentile(eda, 25), "eda_iqr")
    eda_start_mean = np.mean(eda[:, :10], axis=1)
    eda_end_mean = np.mean(eda[:, -10:], axis=1)
    add_feat(eda_end_mean - eda_start_mean, "eda_slope_proxy")
    add_feat(np.sum(np.abs(np.diff(eda, axis=1)), axis=1), "eda_line_length")
    add_feat(row_zero_crossings(np.diff(eda, axis=1)), "eda_diff_zcross")

    window = 8
    cs = np.cumsum(eda, axis=1, dtype=np.float32)
    tonic = (cs[:, window:] - cs[:, :-window]) / window
    phasic = eda[:, window:] - tonic
    add_feat(np.sum(phasic**2, axis=1), "eda_phasic_energy")
    add_feat(np.max(phasic, axis=1), "eda_phasic_max")
    add_feat(np.std(phasic, axis=1), "eda_phasic_std")
    add_feat(row_skewness(eda), "eda_skew")
    add_feat(row_kurtosis(eda), "eda_kurt")
    add_feat((np.argmax(eda, axis=1) / 40.0).astype(np.float32), "eda_peak_time")
    add_feat(row_rms(eda), "eda_rms")

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
    np.random.seed(RANDOM_STATE)
    if X_scaled.shape[0] > subsample_size:
        idx = np.random.choice(X_scaled.shape[0], subsample_size, replace=False)
        X_sub, y_sub = X_scaled[idx], y[idx]
    else:
        X_sub, y_sub = X_scaled, y

    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER, tol=LASSO_TOL, random_state=RANDOM_STATE, precompute=True)
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

    train_df = pd.read_csv(train_path)
    feature_columns = [c for c in train_df.columns if c != "hr"]
    if len(feature_columns) != EXPECTED_RAW_FEATURES:
        raise ValueError(f"Expected {EXPECTED_RAW_FEATURES} raw features, got {len(feature_columns)}")
    y_train = train_df["hr"].to_numpy(dtype=np.float64)
    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    del train_df

    Z_train_base, base_names = extract_base_features(X_train_raw, feature_columns)
    del X_train_raw
    Z_train_poly, poly_names = expand_polynomials(Z_train_base, base_names)
    del Z_train_base

    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train_poly)
    del Z_train_poly
    lasso_idx = lasso_select_subsample(Z_train_scaled, y_train, LASSO_MAX_FEATURES, SUBSAMPLE_SIZE)
    Z_train_selected = Z_train_scaled[:, lasso_idx]
    del Z_train_scaled

    model = SGDRegressor(loss='epsilon_insensitive', epsilon=0.0, penalty='l2', learning_rate='adaptive', eta0=0.01, random_state=RANDOM_STATE, max_iter=5000, tol=1e-4)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid={"alpha": SGD_ALPHAS},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    grid_search.fit(Z_train_selected, y_train)
    best_model = grid_search.best_estimator_

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

    predictions = best_model.predict(Z_test_selected)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Predictions contain NaN/Inf.")
    np.savetxt(predictions_path, predictions, fmt="%.10f")

if __name__ == "__main__":
    main()