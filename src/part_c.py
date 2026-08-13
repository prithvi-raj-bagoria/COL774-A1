import os
import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor


# ============================================================
# Configuration
# ============================================================

HUBER_ALPHA = 1e-5
HUBER_EPSILON = 1.25  
HUBER_MAX_ITER = 1000

EXPECTED_RAW_FEATURES = 1640

# ============================================================
# Basic feature helpers
# ============================================================

def row_mean(x):
    return np.mean(x, axis=1, dtype=np.float32)

def row_std(x):
    return np.std(x, axis=1, dtype=np.float32)

def row_min(x):
    return np.min(x, axis=1)

def row_max(x):
    return np.max(x, axis=1)

def row_range(x):
    return np.ptp(x, axis=1)

def row_rms(x):
    return np.sqrt(np.mean(x * x, axis=1, dtype=np.float32))

def row_slope(x):
    n = x.shape[1]
    t = np.arange(n, dtype=np.float32)
    t = t - np.mean(t)
    denominator = np.sum(t * t)
    return (x @ t) / denominator

def row_skewness(x):
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - mean) / std
    return np.mean(z ** 3, axis=1, dtype=np.float32)

def row_kurtosis(x):
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    std = np.std(x, axis=1, keepdims=True, dtype=np.float32) + 1e-7
    z = (x - mean) / std
    return np.mean(z ** 4, axis=1, dtype=np.float32)

def row_autocorr_lag(x, lag):
    if x.shape[1] <= lag:
        return np.zeros(x.shape[0], dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    centered = x - mean
    numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
    denominator = np.sum(centered * centered, axis=1) + 1e-10
    return (numerator / denominator).astype(np.float32)

def row_zero_crossings(x):
    centered = x - np.mean(x, axis=1, keepdims=True, dtype=np.float32)
    return np.sum((centered[:, :-1] * centered[:, 1:]) < 0, axis=1).astype(np.float32)

def row_local_extrema(x):
    diff = np.diff(x, axis=1)
    return np.sum((diff[:, :-1] * diff[:, 1:]) < 0, axis=1).astype(np.float32)

def robust_clip_rows(x, k=6.0):
    median = np.median(x, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    iqr = q75 - q25
    spread = np.where(iqr < 1e-6, 1.0, iqr)
    low = median - k * spread
    high = median + k * spread
    return np.clip(x, low, high).astype(np.float32)

def robust_normalize_rows(x):
    median = np.median(x, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    iqr = q75 - q25
    scale = np.where(iqr < 1e-6, 1.0, iqr)
    normalized = (x - median) / scale
    return np.clip(normalized, -12.0, 12.0).astype(np.float32)

def moving_average_rows(x, window):
    if window <= 1:
        return x.astype(np.float32, copy=True)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(x, ((0, 0), (left, right)), mode="edge")
    csum = np.cumsum(padded, axis=1, dtype=np.float32)
    csum = np.concatenate([np.zeros((x.shape[0], 1), dtype=np.float32), csum], axis=1)
    return ((csum[:, window:] - csum[:, :-window]) / float(window)).astype(np.float32)

def row_lagged_corr(a, b, lag):
    if lag > 0:
        aa = a[:, lag:]
        bb = b[:, :-lag]
    elif lag < 0:
        k = -lag
        aa = a[:, :-k]
        bb = b[:, k:]
    else:
        aa = a
        bb = b
    aa_c = aa - np.mean(aa, axis=1, keepdims=True, dtype=np.float32)
    bb_c = bb - np.mean(bb, axis=1, keepdims=True, dtype=np.float32)
    num = np.sum(aa_c * bb_c, axis=1)
    den = np.sqrt(np.sum(aa_c * aa_c, axis=1) * np.sum(bb_c * bb_c, axis=1)) + 1e-8
    return (num / den).astype(np.float32)


# ============================================================
# Feature assembly helpers
# ============================================================

def add_feature(features, names, values, name):
    features.append(np.asarray(values, dtype=np.float32))
    names.append(name)

def add_percentiles(features, names, x, prefix):
    x_sorted = np.sort(x, axis=1)
    n = x.shape[1]
    
    idx_10 = int(0.10 * (n - 1))
    idx_25 = int(0.25 * (n - 1))
    idx_75 = int(0.75 * (n - 1))
    idx_90 = int(0.90 * (n - 1))

    add_feature(features, names, x_sorted[:, idx_10], f"{prefix}_p10")
    add_feature(features, names, x_sorted[:, idx_25], f"{prefix}_p25")
    add_feature(features, names, x_sorted[:, idx_75], f"{prefix}_p75")
    add_feature(features, names, x_sorted[:, idx_90], f"{prefix}_p90")

def add_basic_features(features, names, x, prefix):
    add_feature(features, names, row_mean(x), f"{prefix}_mean")
    add_feature(features, names, row_std(x), f"{prefix}_std")
    add_feature(features, names, row_min(x), f"{prefix}_min")
    add_feature(features, names, row_max(x), f"{prefix}_max")
    add_feature(features, names, row_range(x), f"{prefix}_range")
    add_feature(features, names, row_rms(x), f"{prefix}_rms")
    add_feature(features, names, row_slope(x), f"{prefix}_slope")

def add_deep_features(features, names, x, prefix, autocorr_lags=None):
    add_basic_features(features, names, x, prefix)
    add_percentiles(features, names, x, prefix)
    add_feature(features, names, row_skewness(x), f"{prefix}_skew")
    add_feature(features, names, row_kurtosis(x), f"{prefix}_kurt")

    if autocorr_lags is not None:
        for lag in autocorr_lags:
            add_feature(features, names, row_autocorr_lag(x, lag), f"{prefix}_autocorr_{lag}")

    add_feature(features, names, row_zero_crossings(x), f"{prefix}_zero_crossings")
    add_feature(features, names, row_local_extrema(x), f"{prefix}_local_extrema")

def add_block_summary(features, names, x, block_size, prefix):
    n_samples = x.shape[1]
    if n_samples % block_size != 0:
        raise ValueError(f"{prefix}: signal length {n_samples} not divisible by {block_size}")

    n_blocks = n_samples // block_size
    blocks = x.reshape(x.shape[0], n_blocks, block_size)
    means = np.mean(blocks, axis=2, dtype=np.float32)
    stds = np.std(blocks, axis=2, dtype=np.float32)

    add_feature(features, names, np.mean(means, axis=1), f"{prefix}_blockmean_mean")
    add_feature(features, names, np.std(means, axis=1), f"{prefix}_blockmean_std")
    add_feature(features, names, row_slope(means), f"{prefix}_blockmean_slope")
    add_feature(features, names, np.mean(stds, axis=1), f"{prefix}_blockstd_mean")


# ============================================================
# SPECTRAL FEATURES
# ============================================================

def add_spectral_features(features, names, bvp, acc_sq):
    bvp_w = bvp * np.hanning(bvp.shape[1])
    acc_w = acc_sq * np.hanning(acc_sq.shape[1])
    
    bvp_pad_len = bvp.shape[1] * 10
    acc_pad_len = acc_sq.shape[1] * 10

    # 1. BVP Spectral 
    bvp_fft = np.abs(np.fft.rfft(bvp_w, n=bvp_pad_len, axis=1))
    bvp_power = bvp_fft ** 2
    bvp_freqs = np.fft.rfftfreq(bvp_pad_len, d=1.0/64.0)
    
    bvp_mask = (bvp_freqs >= 0.7) & (bvp_freqs <= 3.0)
    bvp_hr_fft = bvp_fft[:, bvp_mask]
    bvp_hr_power = bvp_power[:, bvp_mask]
    bvp_hr_freqs = bvp_freqs[bvp_mask]
    
    bvp_peak_idx = np.argmax(bvp_hr_fft, axis=1)
    bvp_peak_bpm = bvp_hr_freqs[bvp_peak_idx] * 60.0
    add_feature(features, names, bvp_peak_bpm, "bvp_spectral_peak_bpm_fine")
    
    p_bvp = bvp_hr_power / (np.sum(bvp_hr_power, axis=1, keepdims=True) + 1e-10)
    bvp_entropy = -np.sum(p_bvp * np.log2(p_bvp + 1e-10), axis=1)
    add_feature(features, names, bvp_entropy, "bvp_spectral_entropy")
    
    # 2. ACC Spectral 
    acc_fft = np.abs(np.fft.rfft(acc_w, n=acc_pad_len, axis=1))
    acc_freqs = np.fft.rfftfreq(acc_pad_len, d=1.0/32.0)
    
    acc_mask = (acc_freqs >= 0.7) & (acc_freqs <= 3.0)
    acc_hr_fft = acc_fft[:, acc_mask]
    acc_hr_freqs = acc_freqs[acc_mask]
    
    acc_peak_idx = np.argmax(acc_hr_fft, axis=1)
    acc_peak_bpm = acc_hr_freqs[acc_peak_idx] * 60.0
    add_feature(features, names, acc_peak_bpm, "acc_spectral_peak_bpm_fine")
    
    add_feature(features, names, np.abs(bvp_peak_bpm - acc_peak_bpm), "bvp_acc_spectral_diff_fine")

def add_snr_features(features, names, bvp, acc_sq, bvp_bpm_estimate):
    acc_var = np.var(acc_sq, axis=1)
    bvp_var = np.var(bvp, axis=1)
    
    snr = bvp_var / (acc_var + 1e-5)
    add_feature(features, names, snr, "bvp_acc_snr")
    
    confidence = 1.0 / (1.0 + acc_var)
    gated_bpm = bvp_bpm_estimate * confidence
    add_feature(features, names, gated_bpm, "bvp_bpm_gated_by_acc")

def add_eda_derivatives(features, names, eda):
    eda_vel = np.diff(eda, axis=1)
    eda_acc = np.diff(eda_vel, axis=1)
    
    add_feature(features, names, np.mean(eda_vel, axis=1), "eda_velocity_mean")
    add_feature(features, names, np.std(eda_vel, axis=1), "eda_velocity_std")
    add_feature(features, names, np.max(eda_vel, axis=1), "eda_velocity_max")
    
    add_feature(features, names, np.mean(eda_acc, axis=1), "eda_acceleration_mean")
    add_feature(features, names, np.std(eda_acc, axis=1), "eda_acceleration_std")

def add_motion_physics_features(features, names, acc_x, acc_y, acc_z, bvp):
    gravity_x = moving_average_rows(acc_x, 33)
    gravity_y = moving_average_rows(acc_y, 33)
    gravity_z = moving_average_rows(acc_z, 33)
    dynamic_x = acc_x - gravity_x
    dynamic_y = acc_y - gravity_y
    dynamic_z = acc_z - gravity_z

    acc_mag = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)
    gravity_mag = np.sqrt(gravity_x * gravity_x + gravity_y * gravity_y + gravity_z * gravity_z)
    dynamic_mag = np.sqrt(dynamic_x * dynamic_x + dynamic_y * dynamic_y + dynamic_z * dynamic_z)

    add_basic_features(features, names, gravity_mag, "gravity_mag")
    add_deep_features(features, names, dynamic_mag, "dynamic_mag", [1, 2, 4, 8, 16])

    dynamic_jerk = np.diff(dynamic_mag, axis=1)
    add_feature(features, names, np.mean(dynamic_jerk * dynamic_jerk, axis=1), "dynamic_jerk_energy")
    add_feature(features, names, np.mean(np.abs(dynamic_jerk), axis=1), "dynamic_jerk_abs_mean")

    acc_w = dynamic_mag * np.hanning(dynamic_mag.shape[1])
    acc_fft = np.abs(np.fft.rfft(acc_w, axis=1))
    acc_power = acc_fft * acc_fft
    acc_freqs = np.fft.rfftfreq(dynamic_mag.shape[1], d=1.0 / 32.0)
    move_mask = (acc_freqs >= 0.5) & (acc_freqs <= 5.0)
    move_fft = acc_fft[:, move_mask]
    move_power = acc_power[:, move_mask]
    move_freqs = acc_freqs[move_mask]

    peak_idx = np.argmax(move_fft, axis=1)
    move_peak_hz = move_freqs[peak_idx]
    move_peak_power = move_power[np.arange(move_power.shape[0]), peak_idx]
    total_move_power = np.sum(move_power, axis=1) + 1e-10
    add_feature(features, names, move_peak_hz * 60.0, "motion_peak_rate_bpm")
    add_feature(features, names, move_peak_power / total_move_power, "motion_periodicity_strength")

    p_move = move_power / total_move_power[:, None]
    move_entropy = -np.sum(p_move * np.log2(p_move + 1e-10), axis=1)
    add_feature(features, names, move_entropy, "motion_spectral_entropy")

    acc64 = np.repeat(dynamic_mag, 2, axis=1)
    if acc64.shape[1] < bvp.shape[1]:
        pad = bvp.shape[1] - acc64.shape[1]
        acc64 = np.pad(acc64, ((0, 0), (0, pad)), mode="edge")
    elif acc64.shape[1] > bvp.shape[1]:
        acc64 = acc64[:, :bvp.shape[1]]
    lag_candidates = np.asarray([-32, -16, 0, 16, 32], dtype=np.int32)
    lag_corrs = np.column_stack([row_lagged_corr(bvp, acc64, int(lag)) for lag in lag_candidates])
    best_lag_idx = np.argmax(np.abs(lag_corrs), axis=1)
    best_corr = lag_corrs[np.arange(bvp.shape[0]), best_lag_idx]
    best_lag_s = lag_candidates[best_lag_idx] / 64.0
    add_feature(features, names, best_corr, "bvp_acc_lagged_corr")
    add_feature(features, names, best_lag_s, "bvp_acc_lagged_corr_lag_seconds")
    add_feature(features, names, np.mean(np.abs(lag_corrs), axis=1), "bvp_acc_lagged_corr_absmean")

    return {
        "dynamic_mag": dynamic_mag,
        "gravity_mag": gravity_mag,
        "motion_periodicity": (move_peak_power / total_move_power).astype(np.float32),
    }

def _safe_summary(values):
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr)), float(np.std(arr))

def _extract_bvp_row_metrics(signal, fs=64.0):
    x = signal.astype(np.float32)
    n = x.shape[0]
    if n < 8:
        return {
            "sdnn": 0.0, "rmssd": 0.0, "pnn20": 0.0, "pnn50": 0.0, "rr_iqr": 0.0,
            "beat_density": 0.0, "beat_quality": 0.0, "rise_time_mean": 0.0, "rise_time_std": 0.0,
            "decay_time_mean": 0.0, "decay_time_std": 0.0, "systolic_width_mean": 0.0,
            "systolic_width_std": 0.0, "notch_proxy_mean": 0.0, "notch_proxy_std": 0.0,
            "ibi_mean": 0.0, "ibi_std": 0.0
        }

    smooth = moving_average_rows(x[None, :], 3)[0]
    candidates = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:]))[0] + 1
    thr = np.percentile(smooth, 60.0)
    candidates = candidates[smooth[candidates] >= thr]
    min_dist = int(0.30 * fs)
    peaks = []
    last = -10**9
    for idx in candidates:
        if idx - last < min_dist:
            if peaks and smooth[idx] > smooth[peaks[-1]]:
                peaks[-1] = int(idx)
                last = int(idx)
            continue
        peaks.append(int(idx))
        last = int(idx)
    peaks = np.asarray(peaks, dtype=np.int32)

    duration = n / fs
    beat_density = float(len(peaks) / max(duration, 1e-6))
    if len(peaks) < 3:
        return {
            "sdnn": 0.0, "rmssd": 0.0, "pnn20": 0.0, "pnn50": 0.0, "rr_iqr": 0.0,
            "beat_density": beat_density, "beat_quality": 0.0, "rise_time_mean": 0.0, "rise_time_std": 0.0,
            "decay_time_mean": 0.0, "decay_time_std": 0.0, "systolic_width_mean": 0.0,
            "systolic_width_std": 0.0, "notch_proxy_mean": 0.0, "notch_proxy_std": 0.0,
            "ibi_mean": 0.0, "ibi_std": 0.0
        }

    ibis = np.diff(peaks) / fs
    ibi_diff = np.diff(ibis)
    sdnn = float(np.std(ibis) * 1000.0)
    rmssd = float(np.sqrt(np.mean(ibi_diff * ibi_diff)) * 1000.0) if len(ibi_diff) > 0 else 0.0
    pnn20 = float(np.mean(np.abs(ibi_diff) > 0.020)) if len(ibi_diff) > 0 else 0.0
    pnn50 = float(np.mean(np.abs(ibi_diff) > 0.050)) if len(ibi_diff) > 0 else 0.0
    rr_iqr = float((np.percentile(ibis, 75) - np.percentile(ibis, 25)) * 1000.0)
    ibi_mean = float(np.mean(ibis))
    ibi_std = float(np.std(ibis))

    rise_times, decay_times, widths, notch_proxy = [], [], [], []
    valid_beats = 0
    for k in range(1, len(peaks) - 1):
        prev_peak = peaks[k - 1]
        peak = peaks[k]
        next_peak = peaks[k + 1]
        left_segment = smooth[prev_peak:peak + 1]
        right_segment = smooth[peak:next_peak + 1]
        if left_segment.size < 2 or right_segment.size < 2:
            continue

        left_min_rel = int(np.argmin(left_segment))
        right_min_rel = int(np.argmin(right_segment))
        left_min_idx = prev_peak + left_min_rel
        right_min_idx = peak + right_min_rel
        if left_min_idx >= peak or right_min_idx <= peak:
            continue

        amp = smooth[peak] - max(smooth[left_min_idx], smooth[right_min_idx])
        if amp <= 1e-6:
            continue

        rise_times.append((peak - left_min_idx) / fs)
        decay_times.append((right_min_idx - peak) / fs)

        half_level = smooth[left_min_idx] + 0.5 * (smooth[peak] - smooth[left_min_idx])
        up_idx = left_min_idx
        while up_idx < peak and smooth[up_idx] < half_level:
            up_idx += 1
        down_idx = peak
        while down_idx < right_min_idx and smooth[down_idx] > half_level:
            down_idx += 1
        widths.append((down_idx - up_idx) / fs)

        decay_seg = smooth[peak:right_min_idx + 1]
        if decay_seg.size >= 5:
            local_max = np.where((decay_seg[1:-1] > decay_seg[:-2]) & (decay_seg[1:-1] >= decay_seg[2:]))[0] + 1
            if local_max.size > 0:
                notch_amp = decay_seg[local_max[0]] - np.min(decay_seg[local_max[0]:])
                notch_proxy.append(max(0.0, notch_amp / (amp + 1e-6)))
            else:
                notch_proxy.append(0.0)
        else:
            notch_proxy.append(0.0)
        valid_beats += 1

    rise_mean, rise_std = _safe_summary(rise_times)
    decay_mean, decay_std = _safe_summary(decay_times)
    width_mean, width_std = _safe_summary(widths)
    notch_mean, notch_std = _safe_summary(notch_proxy)
    beat_quality = float(valid_beats / max(len(peaks) - 2, 1))

    return {
        "sdnn": sdnn, "rmssd": rmssd, "pnn20": pnn20, "pnn50": pnn50, "rr_iqr": rr_iqr,
        "beat_density": beat_density, "beat_quality": beat_quality,
        "rise_time_mean": rise_mean, "rise_time_std": rise_std,
        "decay_time_mean": decay_mean, "decay_time_std": decay_std,
        "systolic_width_mean": width_mean, "systolic_width_std": width_std,
        "notch_proxy_mean": notch_mean, "notch_proxy_std": notch_std,
        "ibi_mean": ibi_mean, "ibi_std": ibi_std
    }

def add_bvp_biology_features(features, names, bvp):
    metric_names = [
        "sdnn", "rmssd", "pnn20", "pnn50", "rr_iqr",
        "beat_density", "beat_quality", "rise_time_mean", "rise_time_std",
        "decay_time_mean", "decay_time_std", "systolic_width_mean", "systolic_width_std",
        "notch_proxy_mean", "notch_proxy_std", "ibi_mean", "ibi_std"
    ]
    collected = {k: np.zeros(bvp.shape[0], dtype=np.float32) for k in metric_names}
    for i in range(bvp.shape[0]):
        metrics = _extract_bvp_row_metrics(bvp[i], fs=64.0)
        for k in metric_names:
            collected[k][i] = metrics[k]
    for k in metric_names:
        add_feature(features, names, collected[k], f"bvp_{k}")
    return collected

def _extract_eda_row_metrics(signal, fs=4.0):
    tonic = moving_average_rows(signal[None, :], 17)[0]
    phasic = signal - tonic
    phasic_pos = np.maximum(phasic, 0.0)
    peaks = np.where((phasic[1:-1] > phasic[:-2]) & (phasic[1:-1] >= phasic[2:]))[0] + 1
    thr = np.std(phasic) * 0.5 + 1e-4
    peaks = peaks[phasic[peaks] > thr]

    rise_times, recovery_times, amps = [], [], []
    for p in peaks:
        left = p
        while left > 0 and phasic[left] > 0:
            left -= 1
        right = p
        half = 0.5 * phasic[p]
        while right < len(phasic) - 1 and phasic[right] > half:
            right += 1
        rise_times.append((p - left) / fs)
        recovery_times.append((right - p) / fs)
        amps.append(phasic[p])

    rise_mean, rise_std = _safe_summary(rise_times)
    recovery_mean, recovery_std = _safe_summary(recovery_times)
    amp_mean, amp_std = _safe_summary(amps)
    phasic_area = float(np.sum(phasic_pos) / fs)
    tonic_mean = float(np.mean(tonic))
    tonic_slope = float(row_slope(tonic[None, :])[0])

    return {
        "tonic_mean": tonic_mean,
        "tonic_slope": tonic_slope,
        "phasic_mean": float(np.mean(phasic)),
        "phasic_std": float(np.std(phasic)),
        "scr_count": float(len(peaks)),
        "scr_amp_mean": amp_mean,
        "scr_amp_std": amp_std,
        "scr_rise_mean": rise_mean,
        "scr_rise_std": rise_std,
        "scr_recovery_mean": recovery_mean,
        "scr_recovery_std": recovery_std,
        "scr_area": phasic_area,
    }

def add_eda_biology_features(features, names, eda):
    metric_names = [
        "tonic_mean", "tonic_slope", "phasic_mean", "phasic_std", "scr_count",
        "scr_amp_mean", "scr_amp_std", "scr_rise_mean", "scr_rise_std",
        "scr_recovery_mean", "scr_recovery_std", "scr_area"
    ]
    collected = {k: np.zeros(eda.shape[0], dtype=np.float32) for k in metric_names}
    for i in range(eda.shape[0]):
        metrics = _extract_eda_row_metrics(eda[i], fs=4.0)
        for k in metric_names:
            collected[k][i] = metrics[k]
    for k in metric_names:
        add_feature(features, names, collected[k], f"eda_{k}")
    return collected

def add_cross_physiology_features(features, names, bvp, motion_ctx, bvp_bio, eda_bio):
    amp_env = np.sqrt(moving_average_rows((bvp - np.mean(bvp, axis=1, keepdims=True)) ** 2, 32))
    env_w = amp_env * np.hanning(amp_env.shape[1])
    env_fft = np.abs(np.fft.rfft(env_w, axis=1))
    env_power = env_fft * env_fft
    env_freqs = np.fft.rfftfreq(amp_env.shape[1], d=1.0 / 64.0)
    resp_mask = (env_freqs >= 0.1) & (env_freqs <= 0.5)
    resp_power = env_power[:, resp_mask]
    resp_freqs = env_freqs[resp_mask]
    resp_idx = np.argmax(resp_power, axis=1)
    resp_rate_bpm = resp_freqs[resp_idx] * 60.0
    add_feature(features, names, resp_rate_bpm, "bvp_resp_proxy_bpm")

    hrv_rmssd = bvp_bio["rmssd"]
    hrv_sdnn = bvp_bio["sdnn"]
    phasic_std = eda_bio["phasic_std"]
    scr_count = eda_bio["scr_count"]
    add_feature(features, names, hrv_rmssd * phasic_std, "interaction_hrv_rmssd_eda_phasic")
    add_feature(features, names, hrv_sdnn * scr_count, "interaction_hrv_sdnn_scr_count")

    dynamic_mag = motion_ctx["dynamic_mag"]
    motion_thr = np.median(dynamic_mag, axis=1, keepdims=True)
    low_motion = dynamic_mag <= motion_thr
    high_motion = ~low_motion
    bvp_energy = bvp * bvp
    bvp_idx = np.linspace(0, bvp.shape[1] - 1, dynamic_mag.shape[1]).astype(np.int32)
    bvp_motion_energy = bvp_energy[:, bvp_idx]
    low_count = np.sum(low_motion, axis=1).astype(np.float32)
    high_count = np.sum(high_motion, axis=1).astype(np.float32)
    low_energy = np.sum(bvp_motion_energy * low_motion, axis=1) / (low_count + 1e-6)
    high_energy = np.sum(bvp_motion_energy * high_motion, axis=1) / (high_count + 1e-6)
    add_feature(features, names, low_energy, "bvp_energy_low_motion")
    add_feature(features, names, high_energy, "bvp_energy_high_motion")
    add_feature(features, names, high_energy / (low_energy + 1e-6), "bvp_energy_motion_ratio")
    add_feature(features, names, motion_ctx["motion_periodicity"] * bvp_bio["beat_quality"], "interaction_motion_periodicity_beat_quality")


# ============================================================
# BVP / cardiac features
# ============================================================

def vpg_bpm(bvp):
    vpg = np.diff(bvp, axis=1)
    crossings = ((vpg[:, :-1] <= 0) & (vpg[:, 1:] > 0))
    beat_count = np.sum(crossings, axis=1)
    return (beat_count * 6.0).astype(np.float32)

def dominant_cardiac_period(bvp):
    lags = np.asarray([16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76, 86, 96], dtype=np.int32)
    mean = np.mean(bvp, axis=1, keepdims=True, dtype=np.float32)
    centered = bvp - mean
    denominator = np.sum(centered * centered, axis=1) + 1e-10
    
    autocorrelations = []
    for lag in lags:
        numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        autocorrelations.append(numerator / denominator)
        
    autocorrelations = np.column_stack(autocorrelations)
    best_idx = np.argmax(autocorrelations, axis=1)
    best_lag = lags[best_idx]
    best_autocorr = autocorrelations[np.arange(bvp.shape[0]), best_idx]
    
    bpm = (3840.0 / best_lag)
    return bpm.astype(np.float32), best_autocorr.astype(np.float32)

def add_vpg_features(features, names, bvp):
    vpg = np.diff(bvp, axis=1)
    apg = np.diff(vpg, axis=1)

    add_basic_features(features, names, vpg, "vpg")
    add_basic_features(features, names, apg, "apg")
    add_feature(features, names, vpg_bpm(bvp), "vpg_estimated_bpm")

    dominant_bpm, dominant_ac = dominant_cardiac_period(bvp)
    add_feature(features, names, dominant_bpm, "bvp_dominant_bpm")
    add_feature(features, names, dominant_ac, "bvp_dominant_autocorr")

    positive_energy = np.mean(np.maximum(vpg, 0.0) ** 2, axis=1)
    negative_energy = np.mean(np.minimum(vpg, 0.0) ** 2, axis=1) + 1e-10

    add_feature(features, names, positive_energy / negative_energy, "bvp_rise_fall_energy_ratio")
    add_feature(features, names, np.mean(np.maximum(vpg, 0.0), axis=1), "bvp_rise_strength")
    add_feature(features, names, np.mean(np.abs(np.minimum(vpg, 0.0)), axis=1), "bvp_fall_strength")

def add_temporal_position_features(features, names, x, prefix, n_blocks=4):
    n_samples = x.shape[1]
    if n_samples % n_blocks != 0:
        raise ValueError(f"{prefix}: cannot split {n_samples} samples into {n_blocks} equal blocks.")

    block_size = n_samples // n_blocks
    for k in range(n_blocks):
        start = k * block_size
        end = start + block_size
        block = x[:, start:end]
        suffix = f"{prefix}_t{k + 1}"
        
        add_feature(features, names, row_mean(block), f"{suffix}_mean")
        add_feature(features, names, row_std(block), f"{suffix}_std")
        add_feature(features, names, row_slope(block), f"{suffix}_slope")


# ============================================================
# Complete feature extraction
# ============================================================

def extract_features(X_raw, feature_columns):
    features = []
    names = []

    acc_x_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_x_")]
    acc_y_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_y_")]
    acc_z_idx = [i for i, c in enumerate(feature_columns) if c.startswith("acc_z_")]
    bvp_idx = [i for i, c in enumerate(feature_columns) if c.startswith("bvp_")]
    eda_idx = [i for i, c in enumerate(feature_columns) if c.startswith("eda_")]

    acc_x_raw = X_raw[:, acc_x_idx]
    acc_y_raw = X_raw[:, acc_y_idx]
    acc_z_raw = X_raw[:, acc_z_idx]
    bvp_raw = X_raw[:, bvp_idx]
    eda_raw = X_raw[:, eda_idx]

    acc_x = robust_normalize_rows(robust_clip_rows(acc_x_raw))
    acc_y = robust_normalize_rows(robust_clip_rows(acc_y_raw))
    acc_z = robust_normalize_rows(robust_clip_rows(acc_z_raw))
    bvp = robust_normalize_rows(robust_clip_rows(bvp_raw))
    eda = robust_normalize_rows(robust_clip_rows(eda_raw))

    cardiac_lags = [16, 20, 24, 28, 32, 38, 44, 50, 58, 66, 76, 86, 96]
    add_deep_features(features, names, bvp, "bvp", cardiac_lags)
    add_block_summary(features, names, bvp, 64, "bvp")
    add_vpg_features(features, names, bvp)
    add_temporal_position_features(features, names, bvp, "bvp")

    add_basic_features(features, names, acc_x, "acc_x")
    add_basic_features(features, names, acc_y, "acc_y")
    add_basic_features(features, names, acc_z, "acc_z")

    acc_sq = (acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)
    acc_sq_mean = np.mean(acc_sq, axis=1, keepdims=True)
    acc_sq_norm = acc_sq / np.where(acc_sq_mean < 1e-7, 1.0, acc_sq_mean)

    add_deep_features(features, names, acc_sq, "acc_sq", [1, 2, 4, 8, 16])
    add_basic_features(features, names, acc_sq_norm, "acc_sq_norm")
    add_block_summary(features, names, acc_sq, 32, "acc_sq")

    jerk = np.diff(acc_sq, axis=1)
    add_basic_features(features, names, jerk, "jerk")

    acc_sq_sorted = np.sort(acc_sq, axis=1)
    idx_10 = int(0.10 * (acc_sq.shape[1] - 1))
    idx_90 = int(0.90 * (acc_sq.shape[1] - 1))
    burstiness = acc_sq_sorted[:, idx_90] - acc_sq_sorted[:, idx_10]
    add_feature(features, names, burstiness, "acc_sq_burstiness")

    second_diff = np.diff(acc_sq, n=2, axis=1)
    add_feature(features, names, np.mean(np.abs(second_diff), axis=1), "acc_sq_second_diff_abs_mean")
    add_feature(features, names, np.std(second_diff, axis=1), "acc_sq_second_diff_std")
    add_temporal_position_features(features, names, acc_sq, "acc_sq")

    add_deep_features(features, names, eda, "eda", [1, 2, 4])
    add_block_summary(features, names, eda, 4, "eda")
    add_temporal_position_features(features, names, eda, "eda")

    motion_ctx = add_motion_physics_features(features, names, acc_x, acc_y, acc_z, bvp)
    bvp_bio = add_bvp_biology_features(features, names, bvp)
    eda_bio = add_eda_biology_features(features, names, eda)
    add_cross_physiology_features(features, names, bvp, motion_ctx, bvp_bio, eda_bio)

    bvp_bpm_val = vpg_bpm(bvp)
    acc_rms_val = row_rms(acc_sq)
    eda_mean_val = row_mean(eda)

    add_feature(features, names, bvp_bpm_val * acc_rms_val, "interaction_bpm_motion")
    add_feature(features, names, row_std(bvp) * row_std(acc_sq), "interaction_bvp_motion_variability")
    add_feature(features, names, bvp_bpm_val * eda_mean_val, "interaction_bpm_eda")
    add_feature(features, names, acc_rms_val * eda_mean_val, "interaction_motion_eda")

    add_spectral_features(features, names, bvp, acc_sq)
    add_snr_features(features, names, bvp, acc_sq, bvp_bpm_val)
    add_eda_derivatives(features, names, eda)

    Z = np.column_stack(features).astype(np.float32)

    if not np.all(np.isfinite(Z)):
        raise ValueError("Feature matrix contains NaN or Inf.")

    return Z, names

def feature_group_name(feature_name):
    if feature_name.startswith(("gravity_", "dynamic_", "motion_", "bvp_acc_lagged_", "acc_")):
        return "physics_motion"
    if feature_name.startswith(("bvp_", "vpg_", "apg_")):
        return "cardio_biology"
    if feature_name.startswith("eda_"):
        return "eda_biology"
    if feature_name.startswith(("interaction_", "bvp_resp_")):
        return "cross_physiology"
    return "other"

def normalized_mae(y_true, y_pred):
    den = np.abs(y_true - np.mean(y_true)).sum() + 1e-10
    return float(np.abs(y_true - y_pred).sum() / den)

def run_scientific_validation(Z_scaled, y, feature_names):
    n = Z_scaled.shape[0]
    if n < 30:
        print("Skipping scientific validation: not enough samples for stable folds.")
        return

    fold_ids = np.arange(n) % 5
    model_cfg = dict(alpha=HUBER_ALPHA, epsilon=HUBER_EPSILON, max_iter=HUBER_MAX_ITER, tol=1e-3)

    # Feature importance stability across folds
    fold_coefs = []
    fold_scores = []
    for fold in range(5):
        train_mask = fold_ids != fold
        val_mask = ~train_mask
        model = HuberRegressor(**model_cfg)
        model.fit(Z_scaled[train_mask], y[train_mask])
        pred = model.predict(Z_scaled[val_mask])
        fold_scores.append(normalized_mae(y[val_mask], pred))
        fold_coefs.append(model.coef_)
    fold_coefs = np.vstack(fold_coefs)
    coef_cv = np.std(fold_coefs, axis=0) / (np.mean(np.abs(fold_coefs), axis=0) + 1e-6)
    stable_idx = np.argsort(coef_cv)[:10]
    print("\n[Validation] 5-fold NMAE mean: %.6f +/- %.6f" % (np.mean(fold_scores), np.std(fold_scores)))
    print("[Validation] Top stable features (lowest coefficient CV):")
    for idx in stable_idx:
        print("  %s | coef_cv=%.4f" % (feature_names[idx], coef_cv[idx]))

    # Group-wise ablation
    groups = {}
    for i, name in enumerate(feature_names):
        groups.setdefault(feature_group_name(name), []).append(i)

    base_score = float(np.mean(fold_scores))
    print("[Validation] Group ablation vs baseline:")
    for group_name, idxs in sorted(groups.items()):
        keep_mask = np.ones(Z_scaled.shape[1], dtype=bool)
        keep_mask[np.asarray(idxs, dtype=np.int32)] = False
        if np.sum(keep_mask) < 10:
            continue
        scores = []
        for fold in range(5):
            train_mask = fold_ids != fold
            val_mask = ~train_mask
            model = HuberRegressor(**model_cfg)
            model.fit(Z_scaled[train_mask][:, keep_mask], y[train_mask])
            pred = model.predict(Z_scaled[val_mask][:, keep_mask])
            scores.append(normalized_mae(y[val_mask], pred))
        ablated = float(np.mean(scores))
        delta = ablated - base_score
        print("  remove %-18s -> NMAE %.6f (delta %+0.6f)" % (group_name, ablated, delta))


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage:\n"
            "python3 part_c.py train.csv test.csv predictions.txt"
        )

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    predictions_path = sys.argv[3]

    print("Loading training CSV...")
    train_df = pd.read_csv(train_path)

    if "hr" not in train_df.columns:
        raise ValueError("Target column 'hr' not found in training data.")

    feature_columns = [c for c in train_df.columns if c != "hr"]

    if len(feature_columns) != EXPECTED_RAW_FEATURES:
        raise ValueError(f"Expected {EXPECTED_RAW_FEATURES} raw features, got {len(feature_columns)}")

    y_train = train_df["hr"].to_numpy(dtype=np.float64)
    X_train_raw = train_df[feature_columns].to_numpy(dtype=np.float32)
    del train_df

    print("Creating comprehensive domain-engineered features...")
    Z_train, feature_names = extract_features(X_train_raw, feature_columns)
    del X_train_raw

    print(f"Feature matrix built successfully: {Z_train.shape}")
    print(f"Total number of engineered features: {len(feature_names)}")

    print("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    Z_train_scaled = scaler.fit_transform(Z_train)
    del Z_train

    print("\nFitting final HuberRegressor...")
    model = HuberRegressor(
        alpha=HUBER_ALPHA,
        epsilon=HUBER_EPSILON,
        max_iter=HUBER_MAX_ITER,
        tol=1e-3  
    )
    model.fit(Z_train_scaled, y_train)

    print("Huber model fitted successfully.")
    print(f"Iterations used = {model.n_iter_}")

    if os.environ.get("PART_C_VALIDATE", "0") == "1":
        print("\nRunning optional scientific validation (feature stability + ablation)...")
        run_scientific_validation(Z_train_scaled, y_train, feature_names)

    # ========================================================
    # INSPECT COEFFICIENTS TO SEE WHAT THE MODEL USES
    # ========================================================
    coef_importance = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    print("\n--- TOP 15 FEATURES BY ABSOLUTE COEFFICIENT WEIGHT ---")
    for fname, fcoef in coef_importance[:15]:
        print(f"  {fname}: {fcoef:.4f}")
    print("------------------------------------------------------\n")

    del Z_train_scaled
    del y_train

    print("Loading test CSV...")
    test_df = pd.read_csv(test_path)
    X_test_raw = test_df[feature_columns].to_numpy(dtype=np.float32)
    del test_df

    print("Creating test engineered features...")
    Z_test, _ = extract_features(X_test_raw, feature_columns)
    del X_test_raw

    Z_test_scaled = scaler.transform(Z_test)
    del Z_test

    print("Generating predictions...")
    predictions = model.predict(Z_test_scaled)

    np.savetxt(predictions_path, predictions, fmt="%.10f")
    print(f"Successfully saved {len(predictions)} predictions to {predictions_path}")

if __name__ == "__main__":
    main()