import sys
import pickle
import gc
from pathlib import Path
from functools import partial

import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis, iqr

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

FINAL_FEATURES = 495
BASE_SELECT_K = 80
RANDOM_STATE = 42
CV_FOLDS = 3
TEMPORAL_BLOCKS = 6
SGD_ALPHAS = [1e-4, 1e-3, 1e-2]
mi_scorer = partial(mutual_info_regression, n_jobs=4, random_state=RANDOM_STATE)

def add(features, names, values, name):
    features.append(np.asarray(values, dtype=np.float32))
    names.append(name)

def row_mean(x): return np.nanmean(x, axis=1).astype(np.float32)
def row_std(x): return np.nanstd(x, axis=1).astype(np.float32)
def row_min(x): return np.nanmin(x, axis=1).astype(np.float32)
def row_max(x): return np.nanmax(x, axis=1).astype(np.float32)
def row_range(x): return row_max(x) - row_min(x)
def row_iqr(x): return iqr(x, axis=1, nan_policy='omit').astype(np.float32)
def row_median(x): return np.nanmedian(x, axis=1).astype(np.float32)
def row_rms(x): return np.sqrt(np.nanmean(x * x, axis=1)).astype(np.float32)
def row_slope(x):
    n = x.shape[1]
    t = np.arange(n, dtype=np.float32)
    t = t - t.mean()
    xc = np.nan_to_num(x, nan=0.0)
    return (xc @ t / np.sum(t * t)).astype(np.float32)
def row_skew(x): return skew(x, axis=1, bias=False).astype(np.float32)
def row_kurt(x): return kurtosis(x, axis=1, bias=False).astype(np.float32)
def row_abs_diff_mean(x): return np.nanmean(np.abs(np.diff(x, axis=1)), axis=1).astype(np.float32)
def row_zcross(x):
    m = row_mean(x).reshape(-1,1)
    c = x - m
    return np.sum(c[:, :-1] * c[:, 1:] < 0, axis=1).astype(np.float32)
def row_tkeo(x):
    if x.shape[1] < 3:
        return np.zeros(x.shape[0], dtype=np.float32)
    return np.nanmean(x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:], axis=1).astype(np.float32)

def block_means(x, n_blocks=TEMPORAL_BLOCKS):
    n = x.shape[1]
    block_size = n // n_blocks
    means = []
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else n
        means.append(row_mean(x[:, start:end]))
    return np.column_stack(means)

def row_find_peaks(x, fs, distance, height_quantile=0.5):
    all_peaks = []
    n = x.shape[0]
    for i in range(n):
        row = x[i]
        row_clean = np.nan_to_num(row, nan=np.nanmedian(row))
        height_thr = np.percentile(row_clean, height_quantile * 100)
        peaks, _ = signal.find_peaks(
            row_clean,
            distance=max(1, int(distance * fs)),
            height=height_thr
        )
        all_peaks.append(peaks)
    return all_peaks

def rr_features_from_peaks(peaks, fs):
    n = len(peaks)
    mean_rr = np.zeros(n, dtype=np.float32)
    sdnn = np.zeros(n, dtype=np.float32)
    rmssd = np.zeros(n, dtype=np.float32)
    pnn50 = np.zeros(n, dtype=np.float32)
    nn50 = np.zeros(n, dtype=np.float32)
    sd1 = np.zeros(n, dtype=np.float32)
    sd2 = np.zeros(n, dtype=np.float32)

    for i, pk in enumerate(peaks):
        if len(pk) >= 3:
            rr = np.diff(pk) / fs * 1000.0
            mean_rr[i] = np.mean(rr)
            sdnn[i] = np.std(rr)
            if len(rr) >= 2:
                diff_rr = np.diff(rr)
                rmssd[i] = np.sqrt(np.mean(diff_rr ** 2))
                nn50[i] = np.sum(np.abs(diff_rr) > 50)
                pnn50[i] = nn50[i] / len(diff_rr) * 100.0
                sd1[i] = np.sqrt(0.5 * np.var(diff_rr))
                sd2[i] = np.sqrt(2 * np.var(rr) - 0.5 * np.var(diff_rr))
    return mean_rr, sdnn, rmssd, pnn50, nn50, sd1, sd2

def hrv_frequency_features_from_peaks(peaks, fs):
    n = len(peaks)
    vlf = np.zeros(n, dtype=np.float32)
    lf = np.zeros(n, dtype=np.float32)
    hf = np.zeros(n, dtype=np.float32)
    lf_hf = np.zeros(n, dtype=np.float32)
    total_power = np.zeros(n, dtype=np.float32)

    for i, pk in enumerate(peaks):
        if len(pk) >= 8:
            rr = np.diff(pk) / fs
            rr_time = np.cumsum(rr) - rr[0]
            rr_interp = np.interp(
                np.linspace(rr_time[0], rr_time[-1], 256),
                rr_time,
                rr
            )
            rr_interp = rr_interp - np.mean(rr_interp)
            freqs, psd = signal.welch(rr_interp, fs=4.0, nperseg=64)

            vlf_band = (freqs >= 0.0033) & (freqs < 0.04)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs < 0.4)

            vlf[i] = np.trapezoid(psd[vlf_band], freqs[vlf_band])
            lf[i] = np.trapezoid(psd[lf_band], freqs[lf_band])
            hf[i] = np.trapezoid(psd[hf_band], freqs[hf_band])
            lf_hf[i] = lf[i] / (hf[i] + 1e-10)
            total_power[i] = vlf[i] + lf[i] + hf[i]
    return vlf, lf, hf, lf_hf, total_power

def tonic_phasic_decompose(eda, fs=4.0):
    nyq = fs / 2.0
    b, a = signal.butter(4, 0.05 / nyq, btype='lowpass')
    tonic = signal.filtfilt(b, a, eda, axis=1)
    phasic = eda - tonic
    return tonic, phasic

def extract_features(filepath):
    features, names = [], []
    with np.load(filepath, allow_pickle=False) as data:
        y = data["glucose"] if "glucose" in data.files else None

        temp = data["e4_temp"].astype(np.float32, copy=False).reshape(data["e4_temp"].shape[0], -1)
        eda = data["e4_eda"].astype(np.float32, copy=False).reshape(data["e4_eda"].shape[0], -1)
        hr = data["e4_hr"].astype(np.float32, copy=False).reshape(data["e4_hr"].shape[0], -1)
        bvp = data["e4_bvp"].astype(np.float32, copy=False).reshape(data["e4_bvp"].shape[0], -1)
        acc = data["e4_acc"].astype(np.float32, copy=False).reshape(data["e4_acc"].shape[0], -1, 3)
        ecg = data["zephyr_ecg"].astype(np.float32, copy=False).reshape(data["zephyr_ecg"].shape[0], -1)
        breath = data["zephyr_breathing"].astype(np.float32, copy=False).reshape(data["zephyr_breathing"].shape[0], -1)
        zacc = data["zephyr_acc"].astype(np.float32, copy=False).reshape(data["zephyr_acc"].shape[0], -1, 3)

        acc_mag = np.sqrt(np.sum(acc * acc, axis=2))
        zacc_mag = np.sqrt(np.sum(zacc * zacc, axis=2))

        add(features, names, row_mean(temp), "temp_mean")
        add(features, names, row_std(temp), "temp_std")
        add(features, names, row_min(temp), "temp_min")
        add(features, names, row_max(temp), "temp_max")
        add(features, names, row_range(temp), "temp_range")
        add(features, names, row_iqr(temp), "temp_iqr")
        add(features, names, row_slope(temp), "temp_slope")
        add(features, names, row_skew(temp), "temp_skew")
        add(features, names, row_kurt(temp), "temp_kurt")
        bm_temp = block_means(temp)
        add(features, names, bm_temp[:, -1] - bm_temp[:, 0], "temp_trend")
        add(features, names, row_std(bm_temp), "temp_block_std")

        add(features, names, row_mean(eda), "eda_mean")
        add(features, names, row_std(eda), "eda_std")
        add(features, names, row_min(eda), "eda_min")
        add(features, names, row_max(eda), "eda_max")
        add(features, names, row_range(eda), "eda_range")
        add(features, names, row_iqr(eda), "eda_iqr")
        add(features, names, row_slope(eda), "eda_slope")
        add(features, names, row_skew(eda), "eda_skew")
        add(features, names, row_kurt(eda), "eda_kurt")
        add(features, names, row_abs_diff_mean(eda), "eda_abs_diff_mean")

        tonic, phasic = tonic_phasic_decompose(eda)
        add(features, names, row_mean(tonic), "eda_tonic_mean")
        add(features, names, row_std(tonic), "eda_tonic_std")
        add(features, names, row_slope(tonic), "eda_tonic_slope")
        add(features, names, row_mean(phasic), "eda_phasic_mean")
        add(features, names, row_std(phasic), "eda_phasic_std")
        add(features, names, row_max(phasic), "eda_phasic_max")

        scr_count = np.zeros(eda.shape[0], dtype=np.float32)
        scr_amp_mean = np.zeros(eda.shape[0], dtype=np.float32)
        scr_amp_max = np.zeros(eda.shape[0], dtype=np.float32)
        phasic_energy = np.zeros(eda.shape[0], dtype=np.float32)
        for i in range(eda.shape[0]):
            p = np.nan_to_num(phasic[i], nan=0.0)
            peaks, props = signal.find_peaks(p, distance=100, height=np.percentile(p, 80))
            scr_count[i] = len(peaks)
            if len(peaks) > 0:
                scr_amp_mean[i] = np.mean(props['peak_heights'])
                scr_amp_max[i] = np.max(props['peak_heights'])
            phasic_energy[i] = np.sum(p ** 2)
        add(features, names, scr_count, "eda_scr_count")
        add(features, names, scr_amp_mean, "eda_scr_amp_mean")
        add(features, names, scr_amp_max, "eda_scr_amp_max")
        add(features, names, phasic_energy, "eda_phasic_energy")

        add(features, names, row_mean(hr), "hr_mean")
        add(features, names, row_std(hr), "hr_std")
        add(features, names, row_min(hr), "hr_min")
        add(features, names, row_max(hr), "hr_max")
        add(features, names, row_range(hr), "hr_range")
        add(features, names, row_iqr(hr), "hr_iqr")
        add(features, names, row_slope(hr), "hr_slope")
        add(features, names, row_skew(hr), "hr_skew")
        add(features, names, row_kurt(hr), "hr_kurt")
        add(features, names, row_abs_diff_mean(hr), "hr_abs_diff_mean")
        bm_hr = block_means(hr)
        add(features, names, bm_hr[:, -1] - bm_hr[:, 0], "hr_trend")
        add(features, names, row_std(bm_hr), "hr_block_std")

        ecg_peaks = row_find_peaks(ecg, fs=250.0, distance=0.3, height_quantile=0.7)
        mean_rr_ecg, sdnn_ecg, rmssd_ecg, pnn50_ecg, nn50_ecg, sd1_ecg, sd2_ecg = rr_features_from_peaks(ecg_peaks, fs=250.0)
        vlf_ecg, lf_ecg, hf_ecg, lf_hf_ecg, total_ecg = hrv_frequency_features_from_peaks(ecg_peaks, fs=250.0)
        add(features, names, mean_rr_ecg, "ecg_mean_rr")
        add(features, names, sdnn_ecg, "ecg_sdnn")
        add(features, names, rmssd_ecg, "ecg_rmssd")
        add(features, names, pnn50_ecg, "ecg_pnn50")
        add(features, names, nn50_ecg, "ecg_nn50")
        add(features, names, sd1_ecg, "ecg_sd1")
        add(features, names, sd2_ecg, "ecg_sd2")
        add(features, names, sd1_ecg / (sd2_ecg + 1e-7), "ecg_sd_ratio")
        add(features, names, vlf_ecg, "ecg_vlf_power")
        add(features, names, lf_ecg, "ecg_lf_power")
        add(features, names, hf_ecg, "ecg_hf_power")
        add(features, names, lf_hf_ecg, "ecg_lf_hf")
        add(features, names, total_ecg, "ecg_total_power")
        add(features, names, row_std(ecg), "ecg_std")
        add(features, names, row_range(ecg), "ecg_range")
        add(features, names, row_abs_diff_mean(ecg), "ecg_abs_diff_mean")
        add(features, names, row_zcross(ecg), "ecg_zcross")
        add(features, names, row_tkeo(ecg), "ecg_tkeo")

        bvp_peaks = row_find_peaks(bvp, fs=64.0, distance=0.3, height_quantile=0.7)
        mean_rr_bvp, sdnn_bvp, rmssd_bvp, pnn50_bvp, nn50_bvp, sd1_bvp, sd2_bvp = rr_features_from_peaks(bvp_peaks, fs=64.0)
        vlf_bvp, lf_bvp, hf_bvp, lf_hf_bvp, total_bvp = hrv_frequency_features_from_peaks(bvp_peaks, fs=64.0)
        add(features, names, mean_rr_bvp, "bvp_mean_rr")
        add(features, names, sdnn_bvp, "bvp_sdnn")
        add(features, names, rmssd_bvp, "bvp_rmssd")
        add(features, names, pnn50_bvp, "bvp_pnn50")
        add(features, names, nn50_bvp, "bvp_nn50")
        add(features, names, sd1_bvp, "bvp_sd1")
        add(features, names, sd2_bvp, "bvp_sd2")
        add(features, names, sd1_bvp / (sd2_bvp + 1e-7), "bvp_sd_ratio")
        add(features, names, vlf_bvp, "bvp_vlf_power")
        add(features, names, lf_bvp, "bvp_lf_power")
        add(features, names, hf_bvp, "bvp_hf_power")
        add(features, names, lf_hf_bvp, "bvp_lf_hf")
        add(features, names, total_bvp, "bvp_total_power")
        add(features, names, row_std(bvp), "bvp_std")
        add(features, names, row_range(bvp), "bvp_range")
        add(features, names, row_abs_diff_mean(bvp), "bvp_abs_diff_mean")
        add(features, names, row_zcross(bvp), "bvp_zcross")
        add(features, names, row_tkeo(bvp), "bvp_tkeo")

        breath_peaks = row_find_peaks(breath, fs=25.0, distance=1.0, height_quantile=0.5)
        breath_rate = np.array([len(p) / (breath.shape[1] / 25.0) * 60.0 for p in breath_peaks], dtype=np.float32)
        add(features, names, breath_rate, "breath_rate")
        add(features, names, row_mean(breath), "breath_mean")
        add(features, names, row_std(breath), "breath_std")
        add(features, names, row_min(breath), "breath_min")
        add(features, names, row_max(breath), "breath_max")
        add(features, names, row_range(breath), "breath_range")
        add(features, names, row_iqr(breath), "breath_iqr")
        add(features, names, row_slope(breath), "breath_slope")
        add(features, names, row_skew(breath), "breath_skew")
        add(features, names, row_kurt(breath), "breath_kurt")
        add(features, names, row_abs_diff_mean(breath), "breath_abs_diff_mean")
        add(features, names, row_zcross(breath), "breath_zcross")
        add(features, names, row_tkeo(breath), "breath_tkeo")
        bm_breath = block_means(breath)
        add(features, names, bm_breath[:, -1] - bm_breath[:, 0], "breath_trend")
        add(features, names, row_std(bm_breath), "breath_block_std")

        for prefix, mag in [("acc", acc_mag), ("zacc", zacc_mag)]:
            add(features, names, row_mean(mag), f"{prefix}_mean")
            add(features, names, row_std(mag), f"{prefix}_std")
            add(features, names, row_min(mag), f"{prefix}_min")
            add(features, names, row_max(mag), f"{prefix}_max")
            add(features, names, row_range(mag), f"{prefix}_range")
            add(features, names, row_iqr(mag), f"{prefix}_iqr")
            add(features, names, row_slope(mag), f"{prefix}_slope")
            add(features, names, row_skew(mag), f"{prefix}_skew")
            add(features, names, row_kurt(mag), f"{prefix}_kurt")
            add(features, names, row_abs_diff_mean(mag), f"{prefix}_abs_diff_mean")
            add(features, names, row_zcross(mag), f"{prefix}_zcross")
            add(features, names, row_tkeo(mag), f"{prefix}_tkeo")
            bm_mag = block_means(mag)
            add(features, names, bm_mag[:, -1] - bm_mag[:, 0], f"{prefix}_trend")
            add(features, names, row_std(bm_mag), f"{prefix}_block_std")

        add(features, names, rmssd_ecg * scr_count, "rmssd_x_scr_count")
        add(features, names, sdnn_ecg * phasic_energy, "sdnn_x_phasic_energy")
        add(features, names, rmssd_ecg * breath_rate, "rmssd_x_breath_rate")
        add(features, names, hf_ecg * breath_rate, "ecg_hf_x_breath_rate")
        add(features, names, row_std(hr) * row_mean(acc_mag), "hr_std_x_activity")
        add(features, names, row_mean(eda) * row_mean(temp), "eda_x_temp")
        add(features, names, breath_rate * row_mean(acc_mag), "breath_x_activity")

        del temp, eda, hr, bvp, acc, ecg, breath, zacc, acc_mag, zacc_mag, tonic, phasic
        gc.collect()

    X_file = np.column_stack(features).astype(np.float32)
    if not np.all(np.isfinite(np.nan_to_num(X_file, nan=0.0, posinf=0.0, neginf=0.0))):
        raise ValueError("Feature matrix contains invalid values.")
    return X_file, y, names

def get_poly_names(base_names, poly):
    if hasattr(poly, "get_feature_names_out"):
        return list(poly.get_feature_names_out(input_features=base_names))
    n = len(base_names)
    names = list(base_names)
    for i in range(n):
        for j in range(i, n):
            names.append(f"{base_names[i]}*{base_names[j]}")
    return names

def run_training(protocol, input_dir, output_model_path):
    files = sorted(Path(input_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {input_dir}")

    all_X, all_y = [], []
    feature_names = None

    for filepath in files:
        X_file, y_file, names = extract_features(filepath)
        all_X.append(X_file)
        if y_file is not None:
            all_y.append(y_file)
        if feature_names is None:
            feature_names = names

    Z = np.vstack(all_X)
    y = np.concatenate(all_y)
    del all_X, all_y
    gc.collect()

    if protocol == "d1":
        scorer = mi_scorer
        pipeline_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select1", SelectKBest(score_func=scorer, k=min(BASE_SELECT_K, Z.shape[1]))),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("select2", SelectKBest(score_func=scorer, k=FINAL_FEATURES)),
            ("model", SGDRegressor(
                loss='epsilon_insensitive',
                epsilon=0.0,
                penalty='l2',
                max_iter=2000,
                random_state=RANDOM_STATE
            ))
        ]
        param_grid = {"model__alpha": SGD_ALPHAS}
    else:
        scorer = partial(f_regression)
        pipeline_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select1", SelectKBest(score_func=scorer, k=min(BASE_SELECT_K, Z.shape[1]))),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler_poly", StandardScaler()),
            ("select2", SelectKBest(score_func=scorer, k=FINAL_FEATURES)),
            ("model", SGDRegressor(
                loss='epsilon_insensitive',
                epsilon=5.0,
                penalty='elasticnet',
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]
        param_grid = {
            "model__alpha": [1e-3, 1e-2, 1e-1],
            "model__l1_ratio": [0.15, 0.50, 0.85]
        }

    pipeline = Pipeline(pipeline_steps)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1
    )
    grid.fit(Z, y)
    model = grid.best_estimator_

    select1 = model.named_steps["select1"]
    select2 = model.named_steps["select2"]
    poly = model.named_steps["poly"]

    base_idx = select1.get_support(indices=True)
    selected_base_names = [feature_names[i] for i in base_idx]

    poly_names_all = get_poly_names(selected_base_names, poly)

    final_idx = select2.get_support(indices=True)
    final_names = [poly_names_all[i] for i in final_idx]

    final_model = model.named_steps["model"]

    state = {
        "format_version": 1,
        "protocol": protocol,
        "intercept": float(final_model.intercept_[0] if isinstance(final_model.intercept_, np.ndarray) else final_model.intercept_),
        "coef": np.asarray(final_model.coef_, dtype=np.float64).ravel(),
        "feature_names": final_names,
        "preprocessing_state": model
    }

    with open(output_model_path, "wb") as f:
        pickle.dump(state, f)

def run_feature_engineering(protocol, input_dir, model_path, output_path):
    with open(model_path, "rb") as f:
        state = pickle.load(f)

    if state["protocol"] != protocol:
        raise ValueError(f"Protocol mismatch: {state['protocol']}")

    pipeline = state["preprocessing_state"]

    files = sorted(Path(input_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {input_dir}")

    all_X = []
    for filepath in files:
        X_file, _, _ = extract_features(filepath)
        all_X.append(X_file)

    Z = np.vstack(all_X)
    del all_X
    gc.collect()

    Z_test = pipeline[:-1].transform(Z)
    np.save(output_path, Z_test)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit("Usage: ...")
    mode = sys.argv[1].lower()
    protocol = sys.argv[2].lower()
    input_dir = sys.argv[3]
    model_path = sys.argv[4]

    if mode == "train":
        run_training(protocol, input_dir, model_path)
    elif mode == "feature_engineering":
        if len(sys.argv) != 6:
            sys.exit("Feature engineering mode requires an output .npy path.")
        run_feature_engineering(protocol, input_dir, model_path, sys.argv[5])
    else:
        sys.exit("Mode must be 'train' or 'feature_engineering'.")