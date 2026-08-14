import sys, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, SGDRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, KFold

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_FEATURES = 1640
LASSO_ALPHA = 0.005
LASSO_MAX_ITER = 2000
LASSO_TOL = 1e-4
CV_FOLDS = 3
POLY_DEGREE = 2
RANDOM_STATE = 42
BASE_SELECT_K = 100          # number of base features kept before polynomial expansion
LASSO_MAX_FEATURES = 600    # cap on final selected features
SGD_ALPHAS = [1e-4, 1e-3]
SGD_EPSILONS = [0.01, 0.05, 0.1]

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
    """Count local maxima above a small threshold."""
    m = np.mean(bvp, axis=1, keepdims=True, dtype=np.float32)
    s = np.std(bvp, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    thr = m + 0.2 * s
    maxima = ((bvp[:, 1:-1] > bvp[:, :-2]) &
              (bvp[:, 1:-1] > bvp[:, 2:]) &
              (bvp[:, 1:-1] > thr))
    return maxima.sum(axis=1).astype(np.float32)

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

    # ---- BVP features ----
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
        (vpg_std / (bvp_std + 1e-7) + 1e-7),
        "bvp_hjorth_complexity")

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

    # additional robust BVP features
    add(np.max(bvp, axis=1) - np.min(bvp, axis=1), "bvp_range")
    add(np.percentile(bvp, 75, axis=1).astype(np.float32) -
        np.percentile(bvp, 25, axis=1).astype(np.float32), "bvp_iqr")
    med = np.median(bvp, axis=1, keepdims=True)
    add(np.median(np.abs(bvp - med), axis=1), "bvp_mad")
    add(np.sqrt(np.mean(bvp ** 2, axis=1)), "bvp_rms")

    # ---- Accelerometer features ----
    acc_sq = ax ** 2 + ay ** 2 + az ** 2
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

    # ---- EDA features ----
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

    # ---- Temporal context ----
    hb = bvp.shape[1] // 2
    ha = ax.shape[1] // 2
    add(std(bvp[:, :hb]), "bvp_std_h1")
    add(std(bvp[:, hb:]), "bvp_std_h2")
    add(sma(ax[:, :ha], ay[:, :ha], az[:, :ha]), "acc_sma_h1")
    add(sma(ax[:, ha:], ay[:, ha:], az[:, ha:]), "acc_sma_h2")
    add(std(bvp[:, hb:]) / (std(bvp[:, :hb]) + 1e-7), "bvp_std_ratio_h2h1")
    add(sma(ax[:, ha:], ay[:, ha:], az[:, ha:]) /
        (sma(ax[:, :ha], ay[:, :ha], az[:, :ha]) + 1e-7),
        "acc_sma_ratio_h2h1")

    Z = np.column_stack(F).astype(np.float32)
    if not np.all(np.isfinite(Z)):
        raise ValueError("Feature matrix contains NaN/Inf.")
    return Z, N

# ------------------------------------------------------------
# Polynomial expansion (degree 2)
# ------------------------------------------------------------
def polynomial_expand(Z, names, degree=2):
    n, p = Z.shape
    count = p
    if degree >= 2:
        count += p * (p + 1) // 2

    out = np.empty((n, count), dtype=np.float32)
    out_names = list(names)
    out[:, :p] = Z
    col = p

    if degree >= 2:
        for i in range(p):
            zi = Z[:, i]
            for j in range(i, p):
                out[:, col] = zi * Z[:, j]
                out_names.append(f"{names[i]}^2" if i == j
                                 else f"{names[i]}*{names[j]}")
                col += 1

    return out, out_names

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 part_c.py train.csv test.csv predictions.txt")

    train_path, test_path, pred_path = sys.argv[1:4]
    total = time.perf_counter()
    sec("PART (C) — OPTIMIZED LINEAR PIPELINE")

    # 1. Load training data
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

    # 2. Feature engineering
    t = time.perf_counter()
    print("\n[2/6] Feature engineering...")
    Z, names = extract_features(X, cols)
    del X
    print(f"    base_features={Z.shape[1]}")

    # Optional supervised pre-selection of base features
    if BASE_SELECT_K is not None and BASE_SELECT_K < Z.shape[1]:
        select_k = SelectKBest(f_regression, k=BASE_SELECT_K)
        Z = select_k.fit_transform(Z, y)
        selected_idx = select_k.get_support(indices=True)
        names = [names[i] for i in selected_idx]
        print(f"    after SelectKBest(k={BASE_SELECT_K}) base_features={Z.shape[1]}")

    Z, poly_names = polynomial_expand(Z, names, degree=POLY_DEGREE)
    print(f"    expanded_features={Z.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # 3. Scaling + Lasso selection
    t = time.perf_counter()
    print("\n[3/6] Scaling + Lasso selection...")
    scaler = StandardScaler()
    Z = scaler.fit_transform(Z)

    selector = SelectFromModel(
        Lasso(alpha=LASSO_ALPHA,
              max_iter=LASSO_MAX_ITER,
              tol=LASSO_TOL,
              random_state=RANDOM_STATE),
        max_features=LASSO_MAX_FEATURES,
        threshold=-np.inf
    )
    Z = selector.fit_transform(Z, y)
    print(f"    selected_features={Z.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # 4. SGD linear model CV
    t = time.perf_counter()
    print("\n[4/6] SGD linear-model CV...")
    grid = GridSearchCV(
        SGDRegressor(loss="epsilon_insensitive",
                     penalty="l2",
                     max_iter=2000,
                     random_state=RANDOM_STATE),
        {"alpha": SGD_ALPHAS, "epsilon": SGD_EPSILONS},
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        pre_dispatch=4
    )
    grid.fit(Z, y)

    model = grid.best_estimator_
    cv_mae = -grid.best_score_
    print(f"    best_params={grid.best_params_}")
    print(f"    CV_MAE={cv_mae:.6f}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # Training diagnostics
    pred = model.predict(Z)
    mean_y = y.mean()
    train_nmae = np.abs(y - pred).sum() / np.abs(y - mean_y).sum()
    train_nmse = ((y - pred) ** 2).sum() / ((y - mean_y) ** 2).sum()
    print(f"    train_NMAE={train_nmae:.6f}")
    print(f"    train_NMSE={train_nmse:.6f}")
    del pred, Z, y

    # 5. Test feature extraction
    t = time.perf_counter()
    print("\n[5/6] Processing test data...")
    df = pd.read_csv(test_path, dtype=np.float32)
    X = df[cols].to_numpy(dtype=np.float32)
    del df

    Z_test, _ = extract_features(X, cols)
    del X

    if BASE_SELECT_K is not None and BASE_SELECT_K < Z_test.shape[1]:
        Z_test = select_k.transform(Z_test)

    Z_test, _ = polynomial_expand(Z_test, names, degree=POLY_DEGREE)
    Z_test = selector.transform(scaler.transform(Z_test))
    print(f"    test_samples={Z_test.shape[0]:,}")
    print(f"    final_features={Z_test.shape[1]}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # 6. Predict
    t = time.perf_counter()
    print("\n[6/6] Generating predictions...")
    pred = model.predict(Z_test)
    if not np.all(np.isfinite(pred)):
        raise ValueError("Predictions contain NaN/Inf.")
    np.savetxt(pred_path, pred, fmt="%.10f")
    print(f"    saved={len(pred):,}")
    print(f"    file={pred_path}")
    print(f"    done in {time.perf_counter()-t:.2f}s")

    # Summary
    sec("FINAL SUMMARY")
    print(f"Base features after selection: {len(names)}")
    print(f"Expanded features             : {len(poly_names)}")
    print(f"Selected features             : {Z_test.shape[1]}")
    print(f"Best parameters               : {grid.best_params_}")
    print(f"CV MAE                        : {cv_mae:.6f}")
    print(f"Train NMAE                    : {train_nmae:.6f}")
    print(f"Train NMSE                    : {train_nmse:.6f}")
    print(f"Total runtime                 : {time.perf_counter()-total:.2f}s")
    print("\nPublic NMAE/NMSE are computed by the official evaluator.")

if __name__ == "__main__":
    main()