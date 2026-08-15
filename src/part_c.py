import sys, time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_FEATURES = 1640

# Lasso feature selection
LASSO_ALPHA = 0.005
LASSO_MAX_ITER = 1000
LASSO_TOL = 1e-2
LASSO_MAX_FEATURES = 500

# Final SGD model hyperparameters
SGD_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2]
SGD_EPSILONS = [0.01, 0.1, 0.2]
CV_FOLDS = 3
RANDOM_STATE = 42

# ------------------------------------------------------------
# Small mathematical helpers
# ------------------------------------------------------------
def sec(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}")

def mean(x):
    """Mean along axis=1 (per row)."""
    return np.mean(x, axis=1, dtype=np.float64)

def std(x):
    """Standard deviation along axis=1 (per row)."""
    return np.std(x, axis=1, dtype=np.float64)

def skew(x):
    """
    Skewness = E[ ((x - mu) / sigma)^3 ]
    Measures asymmetry of the distribution.
    """
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float64)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float64) + 1e-7
    return np.mean(((x - m) / s) ** 3, axis=1, dtype=np.float64)

def kurt(x):
    """
    Kurtosis = E[ ((x - mu) / sigma)^4 ]
    Measures tailedness / peakedness.
    """
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float64)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float64) + 1e-7
    return np.mean(((x - m) / s) ** 4, axis=1, dtype=np.float64)

def zcross(x):
    """
    Zero crossing rate after mean removal.
    Count of sign changes in centered signal.
    """
    c = x - np.mean(x, axis=1, keepdims=True, dtype=np.float64)
    return np.sum(c[:, :-1] * c[:, 1:] < 0, axis=1).astype(np.float64)

def extrema(x):
    """
    Count local extrema from sign changes of first difference.
    """
    d = np.diff(x, axis=1)
    return np.sum(d[:, :-1] * d[:, 1:] < 0, axis=1).astype(np.float64)

def ac_many(x, lags):
    """
    Autocorrelation at given lags:
        R(k) = sum_t (x_t - mu)(x_{t+k} - mu) / sum_t (x_t - mu)^2
    """
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float64)
    c = x - m
    den = np.sum(c * c, axis=1, dtype=np.float64) + 1e-10
    out = np.empty((x.shape[0], len(lags)), dtype=np.float64)
    for j, lag in enumerate(lags):
        out[:, j] = np.sum(c[:, :-lag] * c[:, lag:], axis=1, dtype=np.float64) / den
    return out

def sma(x, y, z):
    """
    Signal Magnitude Area = sum( |x| + |y| + |z| )
    """
    return np.sum(np.abs(x) + np.abs(y) + np.abs(z), axis=1, dtype=np.float64)

def tkeo(x):
    """
    Teager-Kaiser Energy Operator: x_t^2 - x_{t-1} x_{t+1}
    Averaged over time.
    """
    if x.shape[1] < 3:
        return np.zeros(x.shape[0], dtype=np.float64)
    return np.mean(x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:], axis=1, dtype=np.float64)

def entropy_proxy(x):
    """
    A simple distribution entropy estimator:
    Divide z-scored signal into bins and compute Shannon entropy.
    """
    m = np.mean(x, axis=1, keepdims=True, dtype=np.float64)
    s = np.std(x, axis=1, keepdims=True, dtype=np.float64) + 1e-7
    z = (x - m) / s

    p = np.empty((x.shape[0], 6), dtype=np.float64)
    p[:, 0] = np.mean(z < -2, axis=1)
    p[:, 1] = np.mean((z >= -2) & (z < -1), axis=1)
    p[:, 2] = np.mean((z >= -1) & (z < 0), axis=1)
    p[:, 3] = np.mean((z >= 0) & (z < 1), axis=1)
    p[:, 4] = np.mean((z >= 1) & (z < 2), axis=1)
    p[:, 5] = np.mean(z >= 2, axis=1)

    p += 1e-10
    return -np.sum(p * np.log(p), axis=1).astype(np.float64)

def phasic_energy(x):
    """
    Phasic energy = sum( (x - moving_average)^2 )
    Moving average window w=8 removes slow tonic component.
    """
    w = 8
    if x.shape[1] < w:
        return np.zeros(x.shape[0], dtype=np.float64)
    cs = np.concatenate(
        [np.zeros((x.shape[0], 1), dtype=np.float64),
         np.cumsum(x, axis=1, dtype=np.float64)], axis=1
    )
    tonic = (cs[:, w:] - cs[:, :-w]) / w
    phasic = x[:, w - 1:] - tonic
    return np.sum(phasic ** 2, axis=1, dtype=np.float64)

def peak_count(bvp, fs=64):
    """
    Count local maxima above mean + 0.2*std.
    Used for BPM estimate.
    """
    m = np.mean(bvp, axis=1, keepdims=True, dtype=np.float64)
    s = np.std(bvp, axis=1, keepdims=True, dtype=np.float64) + 1e-7
    thr = m + 0.2 * s
    maxima = ((bvp[:, 1:-1] > bvp[:, :-2]) &
              (bvp[:, 1:-1] > bvp[:, 2:]) &
              (bvp[:, 1:-1] > thr))
    return maxima.sum(axis=1).astype(np.float64)

def rr_features(bvp, fs=64):
    """
    Compute RR interval statistics from detected BVP peaks.
    RR intervals in milliseconds.
    """
    m = np.mean(bvp, axis=1, keepdims=True, dtype=np.float64)
    s = np.std(bvp, axis=1, keepdims=True, dtype=np.float64) + 1e-7
    thr = m + 0.2 * s
    is_peak = ((bvp[:, 1:-1] > bvp[:, :-2]) &
               (bvp[:, 1:-1] > bvp[:, 2:]) &
               (bvp[:, 1:-1] > thr))

    n = bvp.shape[0]
    rr_mean = np.zeros(n, dtype=np.float64)
    rr_std = np.zeros(n, dtype=np.float64)
    rr_rmssd = np.zeros(n, dtype=np.float64)
    rr_median = np.zeros(n, dtype=np.float64)

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

# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
def extract_features(X, cols):
    """
    Convert raw 1640 measurements into manually engineered features.
    All features are computed with NumPy only.
    """
    F, N = [], []

    def add(x, name):
        F.append(np.asarray(x, dtype=np.float64))
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

    # ----------------- BVP features -------------------
    bvp_std = std(bvp)
    add(bvp_std, "bvp_std")
    add(skew(bvp), "bvp_skew")
    add(kurt(bvp), "bvp_kurt")
    add(zcross(bvp), "bvp_zcross")
    add(extrema(bvp), "bvp_extrema")
    add(entropy_proxy(bvp), "bvp_entropy")

    vpg = np.diff(bvp, axis=1)          # first derivative
    apg = np.diff(vpg, axis=1)          # second derivative
    vpg_std = std(vpg)
    apg_std = std(apg)

    add(vpg_std / (bvp_std + 1e-7), "bvp_hjorth_mobility")
    add((apg_std / (vpg_std + 1e-7)) /
        (vpg_std / (bvp_std + 1e-7) + 1e-7), "bvp_hjorth_complexity")

    # Autocorrelation at physiologically plausible heart rates
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

    # BPM from zero crossings of derivative
    bpm_zc = (np.sum((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0), axis=1) * 6.0).astype(np.float64)
    add(bpm_zc, "vpg_est_bpm")
    add(peak_count(bvp) * 6.0, "bvp_peak_bpm")

    # Robust BVP statistics
    add(np.max(bvp, axis=1) - np.min(bvp, axis=1), "bvp_range")
    add(np.percentile(bvp, 75, axis=1).astype(np.float64) -
        np.percentile(bvp, 25, axis=1).astype(np.float64), "bvp_iqr")
    med = np.median(bvp, axis=1, keepdims=True)
    add(np.median(np.abs(bvp - med), axis=1), "bvp_mad")
    add(np.sqrt(np.mean(bvp ** 2, axis=1)), "bvp_rms")

    # RR interval features
    rr_mean, rr_std, rr_rmssd, rr_median = rr_features(bvp)
    add(rr_mean, "bvp_rr_mean")
    add(rr_std, "bvp_rr_std")
    add(rr_rmssd, "bvp_rr_rmssd")
    add(rr_median, "bvp_rr_median")
    hr_from_rr = np.divide(60000.0, rr_mean,
                           out=np.zeros_like(rr_mean),
                           where=rr_mean > 0).astype(np.float64)
    add(hr_from_rr, "bvp_hr_from_rr")

    # ----------------- Accelerometer features -----------------
    acc_sq = ax**2 + ay**2 + az**2
    add(sma(ax, ay, az), "acc_sma")
    add(std(acc_sq), "acc_sq_std")
    add(tkeo(ax) + tkeo(ay) + tkeo(az), "acc_tkeo")

    acc_mag = np.sqrt(acc_sq)
    add(mean(acc_mag), "acc_mag_mean")
    add(std(acc_mag), "acc_mag_std")

    for name, arr in zip(["ax", "ay", "az"], [ax, ay, az]):
        add(mean(arr), f"acc_{name}_mean")
        add(std(arr), f"acc_{name}_std")
        add(skew(arr), f"acc_{name}_skew")
        add(kurt(arr), f"acc_{name}_kurt")

    # ----------------- EDA features -----------------
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

    # ----------------- Temporal context -----------------
    hb = bvp.shape[1] // 2
    ha = ax.shape[1] // 2
    add(std(bvp[:, :hb]), "bvp_std_h1")
    add(std(bvp[:, hb:]), "bvp_std_h2")
    add(sma(ax[:, :ha], ay[:, :ha], az[:, :ha]), "acc_sma_h1")
    add(sma(ax[:, ha:], ay[:, ha:], az[:, ha:]), "acc_sma_h2")
    add(std(bvp[:, hb:]) / (std(bvp[:, :hb]) + 1e-7), "bvp_std_ratio_h2h1")
    add(sma(ax[:, ha:], ay[:, ha:], az[:, ha:]) /
        (sma(ax[:, :ha], ay[:, :ha], az[:, :ha]) + 1e-7), "acc_sma_ratio_h2h1")

    # ----------------- Per‑block aggregates -----------------
    bvp_blocks = bvp.reshape(bvp.shape[0], 10, 64)
    bvp_block_std = np.std(bvp_blocks, axis=2)
    add(np.mean(bvp_block_std, axis=1), "bvp_block_std_mean")
    add(np.std(bvp_block_std, axis=1), "bvp_block_std_std")

    acc = np.stack([ax, ay, az], axis=2)
    acc_blocks = acc.reshape(acc.shape[0], 10, 32, 3)
    acc_block_sma = np.sum(np.abs(acc_blocks), axis=(2, 3))
    add(np.mean(acc_block_sma, axis=1), "acc_block_sma_mean")
    add(np.std(acc_block_sma, axis=1), "acc_block_sma_std")

    Z = np.column_stack(F).astype(np.float64)
    if not np.all(np.isfinite(Z)):
        raise ValueError("Feature matrix contains NaN/Inf.")
    return Z, N

# ------------------------------------------------------------
# Degree-2 polynomial expansion
# ------------------------------------------------------------
def polynomial_expand_deg2(Z, names):
    """
    Creates terms: x_i, x_i*x_j (i <= j), x_i^2.
    Total = p + p*(p+1)/2
    """
    n, p = Z.shape
    count = p + p * (p + 1) // 2
    out = np.empty((n, count), dtype=np.float64)
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
# Main pipeline
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")

    train_path, test_path, pred_path = sys.argv[1:4]
    total = time.perf_counter()
    sec("PART (C) — PIPELINE WITH NO CV LEAKAGE")

    # 1. Load training data
    t = time.perf_counter()
    print("[1/6] Loading training data...")
    df = pd.read_csv(train_path, dtype=np.float64)
    cols = [c for c in df.columns if c != "hr"]
    if len(cols) != RAW_FEATURES:
        raise ValueError(f"Expected {RAW_FEATURES} raw features, got {len(cols)}")
    y = df["hr"].to_numpy(dtype=np.float64)
    X = df[cols].to_numpy(dtype=np.float64)
    del df
    print(f"    samples={len(y):,}, raw_features={X.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # 2. Feature engineering
    t = time.perf_counter()
    print("\n[2/6] Feature engineering...")
    Z, names = extract_features(X, cols)
    del X
    print(f"    base_features={Z.shape[1]}")

    # Degree-2 polynomial expansion
    Z_poly, poly_names = polynomial_expand_deg2(Z, names)
    del Z
    print(f"    expanded_features={Z_poly.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # 3. Build leak-free Pipeline
    #    The pipeline applies scaler, Lasso selection, and SGD model
    #    inside each CV fold. This prevents feature selection and scaling
    #    from seeing validation targets.
    t = time.perf_counter()
    print("\n[3/6] Creating Pipeline + GridSearchCV...")

    selector = SelectFromModel(
        Lasso(alpha=LASSO_ALPHA,
              max_iter=LASSO_MAX_ITER,
              tol=LASSO_TOL,
              random_state=RANDOM_STATE),
        max_features=LASSO_MAX_FEATURES,
        threshold=-np.inf
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", selector),
        ("model", SGDRegressor(
            loss="epsilon_insensitive",
            penalty="l2",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])

    # Grid search for final SGD hyperparameters
    param_grid = {
        "model__alpha": SGD_ALPHAS,
        "model__epsilon": SGD_EPSILONS
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=4,
        pre_dispatch=4,
        verbose=1
    )

    grid.fit(Z_poly, y)

    # Best model is the full pipeline (scaler + selector + SGD)
    best_pipe = grid.best_estimator_
    cv_mae = -grid.best_score_
    print(f"    best_params={grid.best_params_}")
    print(f"    CV_MAE={cv_mae:.6f}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # 4. Training diagnostics (optional)
    t = time.perf_counter()
    print("\n[4/6] Training diagnostics...")
    pred_train = best_pipe.predict(Z_poly)
    mean_y = y.mean()
    train_nmae = np.abs(y - pred_train).sum() / np.abs(y - mean_y).sum()
    train_nmse = ((y - pred_train) ** 2).sum() / ((y - mean_y) ** 2).sum()
    print(f"    train_NMAE={train_nmae:.6f}")
    print(f"    train_NMSE={train_nmse:.6f}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # 5. Test feature extraction and prediction
    t = time.perf_counter()
    print("\n[5/6] Processing test data...")
    df = pd.read_csv(test_path, dtype=np.float64)
    X_test = df[cols].to_numpy(dtype=np.float64)
    del df

    Z_test, _ = extract_features(X_test, cols)
    del X_test
    Z_test_poly, _ = polynomial_expand_deg2(Z_test, names)
    del Z_test

    # The pipeline handles scaling, selection, and prediction on test
    pred = best_pipe.predict(Z_test_poly)

    if not np.all(np.isfinite(pred)):
        raise ValueError("Predictions contain NaN/Inf.")
    np.savetxt(pred_path, pred, fmt="%.10f")
    print(f"    test_samples={Z_test_poly.shape[0]:,}")
    print(f"    final_features={best_pipe.named_steps['selector'].get_support().sum()}")
    print(f"    saved={len(pred):,}")
    print(f"    file={pred_path}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # Summary
    sec("FINAL SUMMARY")
    print(f"Base features                 : {len(names)}")
    print(f"Expanded features             : {len(poly_names)}")
    print(f"Best parameters               : {grid.best_params_}")
    print(f"CV MAE                        : {cv_mae:.6f}")
    print(f"Train NMAE                    : {train_nmae:.6f}")
    print(f"Train NMSE                    : {train_nmse:.6f}")
    print(f"Total runtime                 : {time.perf_counter()-total:.2f}s")
    print("\nPublic NMAE/NMSE are computed by the official evaluator.")

if __name__ == "__main__":
    main()