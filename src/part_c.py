import sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import f_regression

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_FEATURES = 1640
LASSO_ALPHA = 0.01
LASSO_MAX_ITER = 1500
LASSO_TOL = 1e-2
CV_FOLDS = 3
POLY_DEGREE = 2
RANDOM_STATE = 42
BASE_SELECT_K = 50           # if > number of features, no base selection
LASSO_MAX_FEATURES = 500      # top features to keep after Lasso
SGD_ALPHAS = [1e-5, 1e-4]
SGD_EPSILONS = [0.01, 0.1, 0.2]

# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def sec(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}")

def mean(x): return np.mean(x, axis=1, dtype=np.float32)
def std(x): return np.std(x, axis=1, dtype=np.float32)

def skew(x):
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - m) / s) ** 3, axis=1, dtype=np.float32)

def kurt(x):
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    return np.mean(((x - m) / s) ** 4, axis=1, dtype=np.float32)

def zcross(x):
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum(c[:, :-1] * c[:, 1:] < 0, axis=1).astype(np.float32)

def extrema(x):
    d = np.diff(x, axis=1)
    return np.sum(d[:, :-1] * d[:, 1:] < 0, axis=1).astype(np.float32)

def ac_many(x, lags):
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    c = x - m
    den = np.sum(c * c, axis=1, dtype=np.float32) + 1e-10
    out = np.empty((x.shape[0], len(lags)), dtype=np.float32)
    for j, lag in enumerate(lags):
        out[:, j] = np.sum(c[:, :-lag] * c[:, lag:], axis=1, dtype=np.float32) / den
    return out

def sma(x, y, z):
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float32)

def tkeo(x):
    if x.shape[1] < 3:
        return np.zeros(x.shape[0], dtype=np.float32)
    return np.mean(x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:], axis=1, dtype=np.float32)

def entropy_proxy(x):
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - m) / s
    p = np.empty((x.shape[0], 6), dtype=np.float32)
    p[:, 0] = np.mean(z < -2, axis=1)
    p[:, 1] = np.mean((z >= -2) & (z < -1), axis=1)
    p[:, 2] = np.mean((z >= -1) & (z < 0), axis=1)
    p[:, 3] = np.mean((z >= 0) & (z < 1), axis=1)
    p[:, 4] = np.mean((z >= 1) & (z < 2), axis=1)
    p[:, 5] = np.mean(z >= 2, axis=1)
    p += 1e-10
    return -np.sum(p * np.log(p), axis=1).astype(np.float32)

def phasic_energy(x):
    w = 8
    if x.shape[1] < w:
        return np.zeros(x.shape[0], dtype=np.float32)
    cs = np.concatenate([np.zeros((x.shape[0], 1), dtype=np.float32),
                         np.cumsum(x, axis=1, dtype=np.float32)], axis=1)
    tonic = (cs[:, w:] - cs[:, :-w]) / w
    phasic = x[:, w - 1:] - tonic
    return np.sum(phasic ** 2, axis=1, dtype=np.float32)

def peak_count(bvp, fs=64):
    m = np.mean(bvp, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(bvp, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    thr = m + 0.2 * s
    maxima = ((bvp[:, 1:-1] > bvp[:, :-2]) &
              (bvp[:, 1:-1] > bvp[:, 2:]) &
              (bvp[:, 1:-1] > thr))
    return maxima.sum(axis=1).astype(np.float32)

def rr_features(bvp, fs=64):
    m = np.mean(bvp, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(bvp, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    thr = m + 0.2 * s
    is_peak = ((bvp[:, 1:-1] > bvp[:, :-2]) &
               (bvp[:, 1:-1] > bvp[:, 2:]) &
               (bvp[:, 1:-1] > thr))
    n = bvp.shape[0]
    rr_mean = np.zeros(n, dtype=np.float32)
    rr_std = np.zeros(n, dtype=np.float32)
    rr_rmssd = np.zeros(n, dtype=np.float32)
    rr_median = np.zeros(n, dtype=np.float32)
    for i in range(n):
        idx = np.where(is_peak[i])[0] + 1
        if len(idx) >= 2:
            rr = np.diff(idx) / fs * 1000.0
            rr_mean[i] = np.mean(rr)
            rr_std[i] = np.std(rr)
            rr_median[i] = np.median(rr)
            if len(rr) >= 2:
                rr_rmssd[i] = np.sqrt(np.mean(np.diff(rr) ** 2))
    return rr_mean, rr_std, rr_rmssd, rr_median

def peak_detect_robust(bvp, fs=64):
    n = bvp.shape[0]
    peaks_all = []
    for i in range(n):
        x = bvp[i]
        dx = np.diff(x)
        dx = np.concatenate([[0], dx])
        win = 16
        avg_abs_dx = np.convolve(np.abs(dx), np.ones(win)/win, mode='same')
        thr = 0.6 * avg_abs_dx
        is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > thr[1:-1])
        idx = np.where(is_peak)[0] + 1
        if len(idx) > 1:
            filtered = [idx[0]]
            for p in idx[1:]:
                if p - filtered[-1] >= int(0.3 * fs):
                    filtered.append(p)
            idx = np.array(filtered)
        peaks_all.append(idx)
    return peaks_all

def robust_rr_features(bvp, fs=64):
    peaks = peak_detect_robust(bvp, fs)
    n = bvp.shape[0]
    rr_mean = np.zeros(n, dtype=np.float32)
    rr_std = np.zeros(n, dtype=np.float32)
    rr_rmssd = np.zeros(n, dtype=np.float32)
    rr_median = np.zeros(n, dtype=np.float32)
    pulse_amp = np.zeros(n, dtype=np.float32)
    pulse_width = np.zeros(n, dtype=np.float32)
    for i, pk in enumerate(peaks):
        if len(pk) >= 2:
            rr = np.diff(pk) / fs * 1000.0
            rr_mean[i] = np.mean(rr)
            rr_std[i] = np.std(rr)
            rr_median[i] = np.median(rr)
            if len(rr) >= 2:
                rr_rmssd[i] = np.sqrt(np.mean(np.diff(rr) ** 2))
            amps, widths = [], []
            for j in range(len(pk)-1):
                seg = bvp[i, pk[j]:pk[j+1]]
                if len(seg) > 10:
                    peak_val = bvp[i, pk[j]]
                    trough_val = np.min(seg)
                    amps.append(peak_val - trough_val)
                    half = (peak_val + trough_val) / 2
                    above = np.where(bvp[i, pk[j]:pk[j+1]] > half)[0]
                    if len(above) > 0:
                        widths.append((above[-1]-above[0]) / fs * 1000)
            pulse_amp[i] = np.mean(amps) if amps else 0
            pulse_width[i] = np.mean(widths) if widths else 0
    return rr_mean, rr_std, rr_rmssd, rr_median, pulse_amp, pulse_width

def autocorr_hr(bvp, fs=64):
    n = bvp.shape[0]
    hr = np.zeros(n, dtype=np.float32)
    min_lag = int(fs * 60 / 200)
    max_lag = int(fs * 60 / 40)
    for i in range(n):
        x = bvp[i]
        x = x - np.mean(x)
        best_r = -np.inf
        best_lag = 0
        for lag in range(min_lag, max_lag+1):
            if lag >= len(x):
                continue
            r = np.sum(x[:-lag] * x[lag:]) / (np.sqrt(np.sum(x[:-lag]**2)) * np.sqrt(np.sum(x[lag:]**2)) + 1e-10)
            if r > best_r:
                best_r = r
                best_lag = lag
        if best_lag > 0:
            hr[i] = 60.0 * fs / best_lag
    return hr

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def extract_features(X, cols):
    F, N = [], []
    def add(x, name):
        F.append(np.asarray(x, dtype=np.float32))
        N.append(name)

    prefixes = {"ax": "acc_x_", "ay": "acc_y_", "az": "acc_z_",
                "bvp": "bvp_", "eda": "eda_"}
    idx = {k: [i for i, c in enumerate(cols) if c.startswith(p)]
           for k, p in prefixes.items()}

    ax = X[:, idx["ax"]]
    ay = X[:, idx["ay"]]
    az = X[:, idx["az"]]
    bvp = X[:, idx["bvp"]]
    eda = X[:, idx["eda"]]

    # BVP features
    bvp_std = std(bvp)
    add(bvp_std, "bvp_std")
    add(skew(bvp), "bvp_skew")
    add(kurt(bvp), "bvp_kurt")
    add(zcross(bvp), "bvp_zcross")
    add(extrema(bvp), "bvp_extrema")
    add(entropy_proxy(bvp), "bvp_entropy")

    vpg = np.diff(bvp, axis=1)
    apg = np.diff(vpg, axis=1)
    vpg_std = std(vpg)
    apg_std = std(apg)

    add(vpg_std / (bvp_std + 1e-7), "bvp_hjorth_mobility")
    add((apg_std / (vpg_std + 1e-7)) /
        (vpg_std / (bvp_std + 1e-7) + 1e-7), "bvp_hjorth_complexity")

    bpm_targets = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170]
    lags = [int(60 * 64 / bpm) for bpm in bpm_targets]
    ac = ac_many(bvp, lags)
    for j, bpm in enumerate(bpm_targets):
        add(ac[:, j], f"bvp_ac_{bpm}")

    add(zcross(vpg), "vpg_zcross")
    add(zcross(apg), "apg_zcross")
    add(ac[:, 2] + ac[:, 3], "bvp_ac_low")
    add(ac[:, 5] + ac[:, 6], "bvp_ac_high")
    add(vpg_std, "vpg_std")
    add(skew(vpg), "vpg_skew")
    add(apg_std, "apg_std")
    add(skew(apg), "apg_skew")
    add(kurt(apg), "apg_kurt")
    add(extrema(apg), "apg_extrema")

    bpm_zc = (np.sum((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0), axis=1) * 6.0).astype(np.float32)
    add(bpm_zc, "vpg_est_bpm")
    add(peak_count(bvp) * 6.0, "bvp_peak_bpm")

    add(np.max(bvp, axis=1) - np.min(bvp, axis=1), "bvp_range")
    add(np.percentile(bvp, 75, axis=1).astype(np.float32) -
        np.percentile(bvp, 25, axis=1).astype(np.float32), "bvp_iqr")
    med = np.median(bvp, axis=1, keepdims=True)
    add(np.median(np.abs(bvp - med), axis=1), "bvp_mad")
    add(np.sqrt(np.mean(bvp ** 2, axis=1)), "bvp_rms")

    rr_mean, rr_std, rr_rmssd, rr_median = rr_features(bvp)
    add(rr_mean, "bvp_rr_mean")
    add(rr_std, "bvp_rr_std")
    add(rr_rmssd, "bvp_rr_rmssd")
    add(rr_median, "bvp_rr_median")
    hr_from_rr = np.divide(60000.0, rr_mean, out=np.zeros_like(rr_mean), where=rr_mean > 0).astype(np.float32)
    add(hr_from_rr, "bvp_hr_from_rr")

    rr_mean2, rr_std2, rr_rmssd2, rr_median2, pulse_amp, pulse_width = robust_rr_features(bvp)
    add(rr_mean2, "bvp_rr_mean_robust")
    add(rr_std2, "bvp_rr_std_robust")
    add(rr_rmssd2, "bvp_rr_rmssd_robust")
    add(rr_median2, "bvp_rr_median_robust")
    add(pulse_amp, "bvp_pulse_amp_mean")
    add(pulse_width, "bvp_pulse_width_mean")
    hr_from_rr2 = np.divide(60000.0, rr_mean2, out=np.zeros_like(rr_mean2), where=rr_mean2 > 0).astype(np.float32)
    add(hr_from_rr2, "bvp_hr_from_rr_robust")

    hr_auto = autocorr_hr(bvp)
    add(hr_auto, "bvp_hr_autocorr")

    # Accelerometer features
    acc_sq = ax ** 2 + ay ** 2 + az ** 2
    acc_sma_val = sma(ax, ay, az)
    add(acc_sma_val, "acc_sma")
    acc_sq_std_val = std(acc_sq)
    add(acc_sq_std_val, "acc_sq_std")
    add(tkeo(ax) + tkeo(ay) + tkeo(az), "acc_tkeo")

    acc_mag = np.sqrt(acc_sq)
    add(mean(acc_mag), "acc_mag_mean")
    add(std(acc_mag), "acc_mag_std")

    for name, arr in zip(["ax", "ay", "az"], [ax, ay, az]):
        add(mean(arr), f"acc_{name}_mean")
        add(std(arr), f"acc_{name}_std")
        add(skew(arr), f"acc_{name}_skew")
        add(kurt(arr), f"acc_{name}_kurt")

    # EDA features
    add(mean(eda), "eda_mean")
    add(std(eda), "eda_std")
    add(std(np.diff(eda, axis=1)), "eda_diff_std")
    add(phasic_energy(eda), "eda_phasic_energy")
    add(np.min(eda, axis=1), "eda_min")
    add(np.max(eda, axis=1), "eda_max")
    add(np.max(eda, axis=1) - np.min(eda, axis=1), "eda_range")
    add(skew(eda), "eda_skew")
    add(kurt(eda), "eda_kurt")
    add(mean(np.diff(eda, axis=1)), "eda_diff_mean")

    # Temporal context
    hb = bvp.shape[1] // 2
    ha = ax.shape[1] // 2
    add(std(bvp[:, :hb]), "bvp_std_h1")
    add(std(bvp[:, hb:]), "bvp_std_h2")
    add(sma(ax[:, :ha], ay[:, :ha], az[:, :ha]), "acc_sma_h1")
    add(sma(ax[:, ha:], ay[:, ha:], az[:, ha:]), "acc_sma_h2")
    add(std(bvp[:, hb:]) / (std(bvp[:, :hb]) + 1e-7), "bvp_std_ratio_h2h1")
    add(sma(ax[:, ha:], ay[:, ha:], az[:, ha:]) /
        (sma(ax[:, :ha], ay[:, :ha], az[:, :ha]) + 1e-7), "acc_sma_ratio_h2h1")

    # Per-block aggregates
    bvp_blocks = bvp.reshape(bvp.shape[0], 10, 64)
    bvp_block_std = np.std(bvp_blocks, axis=2)
    add(np.mean(bvp_block_std, axis=1), "bvp_block_std_mean")
    add(np.std(bvp_block_std, axis=1), "bvp_block_std_std")

    acc = np.stack([ax, ay, az], axis=2)
    acc_blocks = acc.reshape(acc.shape[0], 10, 32, 3)
    acc_block_sma = np.sum(np.abs(acc_blocks), axis=(2, 3))
    add(np.mean(acc_block_sma, axis=1), "acc_block_sma_mean")
    add(np.std(acc_block_sma, axis=1), "acc_block_sma_std")

    # Motion-artifact interactions
    snr = bvp_std / (acc_sma_val + 1e-7)
    add(snr, "bvp_snr")
    add(rr_std * acc_sma_val, "rr_std_times_sma")
    add(hr_from_rr * acc_sq_std_val, "hr_from_rr_times_acc_sq_std")

    Z = np.column_stack(F).astype(np.float32)
    if not np.all(np.isfinite(Z)):
        raise ValueError("Feature matrix contains NaN/Inf.")
    return Z, N

# ------------------------------------------------------------
# Polynomial expansion (degree 2)
# ------------------------------------------------------------
def polynomial_expand(Z, names):
    n, p = Z.shape
    count = p + p * (p + 1) // 2
    out = np.empty((n, count), dtype=np.float32)
    out_names = list(names)
    out[:, :p] = Z
    col = p
    for i in range(p):
        zi = Z[:, i]
        for j in range(i, p):
            out[:, col] = zi * Z[:, j]
            out_names.append(f"{names[i]}^2" if i == j else f"{names[i]}*{names[j]}")
            col += 1
    if not np.all(np.isfinite(out)):
        raise ValueError("Polynomial matrix contains NaN/Inf.")
    return out, out_names

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")

    train_path, test_path, pred_path = sys.argv[1:4]
    total = time.perf_counter()
    sec("PART (C) — MANUAL FEATURE SELECTION + ROBUST FEATURES")

    t = time.perf_counter()
    print("[1/6] Loading training data...")
    df = pd.read_csv(train_path, dtype=np.float32)
    cols = [c for c in df.columns if c != "hr"]
    if len(cols) != RAW_FEATURES:
        raise ValueError(f"Expected {RAW_FEATURES} raw features, got {len(cols)}")
    y = df["hr"].to_numpy(dtype=np.float64)
    X = df[cols].to_numpy(dtype=np.float32)
    del df
    print(f"    samples={len(y):,}, raw_features={X.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    t = time.perf_counter()
    print("\n[2/6] Feature engineering...")
    Z, names = extract_features(X, cols)
    del X
    print(f"    base_features={Z.shape[1]}")

    # Manual base feature selection using f_regression scores
    if BASE_SELECT_K < Z.shape[1]:
        f_scores, _ = f_regression(Z, y)
        base_idx = np.argsort(f_scores)[-BASE_SELECT_K:]
        base_idx = np.sort(base_idx)
        Z = Z[:, base_idx]
        names = [names[i] for i in base_idx]
        print(f"    after manual f_regression top-k={BASE_SELECT_K} → {Z.shape[1]} base_features")

    Z_poly, poly_names = polynomial_expand(Z, names)
    del Z
    print(f"    expanded_features={Z_poly.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    t = time.perf_counter()
    print("\n[3/6] Scaling + manual Lasso feature selection...")
    scaler = StandardScaler()
    Z_poly = scaler.fit_transform(Z_poly)

    lasso = Lasso(alpha=LASSO_ALPHA,
                  max_iter=LASSO_MAX_ITER,
                  tol=LASSO_TOL,
                  random_state=RANDOM_STATE,
                  precompute=True)
    lasso.fit(Z_poly, y)
    coef_abs = np.abs(lasso.coef_)

    if Z_poly.shape[1] > LASSO_MAX_FEATURES:
        lasso_idx = np.argsort(coef_abs)[-LASSO_MAX_FEATURES:]
        lasso_idx = np.sort(lasso_idx)
    else:
        lasso_idx = np.arange(Z_poly.shape[1])

    Z_selected = Z_poly[:, lasso_idx]
    print(f"    selected_features={Z_selected.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    t = time.perf_counter()
    print("\n[4/6] SGD linear-model CV...")
    grid = GridSearchCV(
        SGDRegressor(loss="epsilon_insensitive", penalty="l2", max_iter=2000, random_state=RANDOM_STATE),
        {"alpha": SGD_ALPHAS, "epsilon": SGD_EPSILONS},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, pre_dispatch=4
    )
    grid.fit(Z_selected, y)
    model = grid.best_estimator_
    cv_mae = -grid.best_score_
    print(f"    best_params={grid.best_params_}")
    print(f"    CV_MAE={cv_mae:.6f}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    pred = model.predict(Z_selected)
    mean_y = y.mean()
    train_nmae = np.abs(y - pred).sum() / np.abs(y - mean_y).sum()
    train_nmse = ((y - pred) ** 2).sum() / ((y - mean_y) ** 2).sum()
    print(f"    train_NMAE={train_nmae:.6f}")
    print(f"    train_NMSE={train_nmse:.6f}")
    del pred, Z_selected, y

    t = time.perf_counter()
    print("\n[5/6] Processing test data...")
    df_test = pd.read_csv(test_path, dtype=np.float32)
    has_labels = 'hr' in df_test.columns
    if has_labels:
        y_test = df_test['hr'].to_numpy(dtype=np.float64)
    else:
        y_test = None
    X_test = df_test[cols].to_numpy(dtype=np.float32)
    del df_test

    Z_test, _ = extract_features(X_test, cols)
    del X_test

    if BASE_SELECT_K < Z_test.shape[1]:
        Z_test = Z_test[:, base_idx]

    Z_test_poly, _ = polynomial_expand(Z_test, names)
    del Z_test
    Z_test_poly = scaler.transform(Z_test_poly)
    Z_test_selected = Z_test_poly[:, lasso_idx]

    pred = model.predict(Z_test_selected)
    if not np.all(np.isfinite(pred)):
        raise ValueError("Predictions contain NaN/Inf.")
    np.savetxt(pred_path, pred, fmt="%.10f")
    print(f"    test_samples={Z_test_selected.shape[0]:,}")
    print(f"    final_features={Z_test_selected.shape[1]}")
    print(f"    saved={len(pred):,}")
    print(f"    file={pred_path}")

    if has_labels:
        mean_y_test = y_test.mean()
        test_nmae = np.abs(y_test - pred).sum() / np.abs(y_test - mean_y_test).sum()
        test_nmse = ((y_test - pred) ** 2).sum() / ((y_test - mean_y_test) ** 2).sum()
        print(f"    Local test NMAE={test_nmae:.6f}")
        print(f"    Local test NMSE={test_nmse:.6f}")

    print(f"    done in {time.perf_counter()-t:.2f}s")

    sec("FINAL SUMMARY")
    print(f"Base features after selection: {len(names)}")
    print(f"Expanded features             : {len(poly_names)}")
    print(f"Selected features             : {Z_test_selected.shape[1]}")
    print(f"Best parameters               : {grid.best_params_}")
    print(f"CV MAE                        : {cv_mae:.6f}")
    print(f"Train NMAE                    : {train_nmae:.6f}")
    print(f"Train NMSE                    : {train_nmse:.6f}")
    if has_labels:
        print(f"Test NMAE                     : {test_nmae:.6f}")
        print(f"Test NMSE                     : {test_nmse:.6f}")
    print(f"Total runtime                 : {time.perf_counter()-total:.2f}s")
    print("\nPublic NMAE/NMSE are computed by the official evaluator.")

if __name__ == "__main__":
    main()