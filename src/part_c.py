import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor


# ============================================================
# Configuration
# ============================================================

HUBER_ALPHA = 1e-5
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 5000

EXPECTED_RAW_FEATURES = 1640
EXPECTED_ENGINEERED_FEATURES = 176


# ============================================================
# Basic feature helpers
# ============================================================

def row_mean(x):
    return np.mean(
        x,
        axis=1,
        dtype=np.float32
    )


def row_std(x):
    return np.std(
        x,
        axis=1,
        dtype=np.float32
    )


def row_min(x):
    return np.min(
        x,
        axis=1
    )


def row_max(x):
    return np.max(
        x,
        axis=1
    )


def row_range(x):
    return np.ptp(
        x,
        axis=1
    )


def row_rms(x):
    return np.sqrt(
        np.mean(
            x * x,
            axis=1,
            dtype=np.float32
        )
    )


def row_slope(x):
    """Least-squares slope of each row against time."""

    n = x.shape[1]

    t = np.arange(
        n,
        dtype=np.float32
    )

    t = t - np.mean(t)

    denominator = np.sum(
        t * t
    )

    return (x @ t) / denominator


def row_skewness(x):
    """Third standardized moment."""

    mean = np.mean(
        x,
        axis=1,
        keepdims=True,
        dtype=np.float32
    )

    std = (
        np.std(
            x,
            axis=1,
            keepdims=True,
            dtype=np.float32
        )
        + 1e-7
    )

    z = (x - mean) / std

    return np.mean(
        z ** 3,
        axis=1,
        dtype=np.float32
    )


def row_kurtosis(x):
    """Fourth standardized moment."""

    mean = np.mean(
        x,
        axis=1,
        keepdims=True,
        dtype=np.float32
    )

    std = (
        np.std(
            x,
            axis=1,
            keepdims=True,
            dtype=np.float32
        )
        + 1e-7
    )

    z = (x - mean) / std

    return np.mean(
        z ** 4,
        axis=1,
        dtype=np.float32
    )


def row_autocorr_lag(x, lag):
    """Normalized autocorrelation at lag k."""

    if x.shape[1] <= lag:
        return np.zeros(
            x.shape[0],
            dtype=np.float32
        )

    mean = np.mean(
        x,
        axis=1,
        keepdims=True,
        dtype=np.float32
    )

    centered = x - mean

    numerator = np.sum(
        centered[:, :-lag]
        * centered[:, lag:],
        axis=1
    )

    denominator = (
        np.sum(
            centered * centered,
            axis=1
        )
        + 1e-10
    )

    return (
        numerator / denominator
    ).astype(np.float32)


def row_zero_crossings(x):
    """Count mean-centered zero crossings."""

    centered = (
        x
        - np.mean(
            x,
            axis=1,
            keepdims=True,
            dtype=np.float32
        )
    )

    return np.sum(
        (
            centered[:, :-1]
            * centered[:, 1:]
        ) < 0,
        axis=1
    ).astype(np.float32)


def row_local_extrema(x):
    """Count local extrema via derivative sign changes."""

    diff = np.diff(
        x,
        axis=1
    )

    return np.sum(
        (
            diff[:, :-1]
            * diff[:, 1:]
        ) < 0,
        axis=1
    ).astype(np.float32)


# ============================================================
# Feature assembly helpers
# ============================================================

def add_feature(
    features,
    names,
    values,
    name
):
    features.append(
        np.asarray(
            values,
            dtype=np.float32
        )
    )

    names.append(name)


def add_percentiles(features, names, x, prefix):
    # Sort each row once instead of calculating percentiles 4 separate times
    x_sorted = np.sort(x, axis=1)
    n = x.shape[1]
    
    # Pre-compute target indices for 10%, 25%, 75%, and 90%
    idx_10 = int(0.10 * (n - 1))
    idx_25 = int(0.25 * (n - 1))
    idx_75 = int(0.75 * (n - 1))
    idx_90 = int(0.90 * (n - 1))

    add_feature(features, names, x_sorted[:, idx_10], f"{prefix}_p10")
    add_feature(features, names, x_sorted[:, idx_25], f"{prefix}_p25")
    add_feature(features, names, x_sorted[:, idx_75], f"{prefix}_p75")
    add_feature(features, names, x_sorted[:, idx_90], f"{prefix}_p90")


def add_basic_features(
    features,
    names,
    x,
    prefix
):
    add_feature(
        features,
        names,
        row_mean(x),
        f"{prefix}_mean"
    )

    add_feature(
        features,
        names,
        row_std(x),
        f"{prefix}_std"
    )

    add_feature(
        features,
        names,
        row_min(x),
        f"{prefix}_min"
    )

    add_feature(
        features,
        names,
        row_max(x),
        f"{prefix}_max"
    )

    add_feature(
        features,
        names,
        row_range(x),
        f"{prefix}_range"
    )

    add_feature(
        features,
        names,
        row_rms(x),
        f"{prefix}_rms"
    )

    add_feature(
        features,
        names,
        row_slope(x),
        f"{prefix}_slope"
    )


def add_deep_features(
    features,
    names,
    x,
    prefix,
    autocorr_lags=None
):
    add_basic_features(
        features,
        names,
        x,
        prefix
    )

    add_percentiles(
        features,
        names,
        x,
        prefix
    )

    add_feature(
        features,
        names,
        row_skewness(x),
        f"{prefix}_skew"
    )

    add_feature(
        features,
        names,
        row_kurtosis(x),
        f"{prefix}_kurt"
    )

    if autocorr_lags is not None:
        for lag in autocorr_lags:
            add_feature(
                features,
                names,
                row_autocorr_lag(
                    x,
                    lag
                ),
                f"{prefix}_autocorr_{lag}"
            )

    add_feature(
        features,
        names,
        row_zero_crossings(x),
        f"{prefix}_zero_crossings"
    )

    add_feature(
        features,
        names,
        row_local_extrema(x),
        f"{prefix}_local_extrema"
    )


def add_block_summary(
    features,
    names,
    x,
    block_size,
    prefix
):
    n_samples = x.shape[1]

    if n_samples % block_size != 0:
        raise ValueError(
            f"{prefix}: signal length {n_samples} "
            f"is not divisible by block size {block_size}"
        )

    n_blocks = (
        n_samples // block_size
    )

    blocks = x.reshape(
        x.shape[0],
        n_blocks,
        block_size
    )

    means = np.mean(
        blocks,
        axis=2,
        dtype=np.float32
    )

    stds = np.std(
        blocks,
        axis=2,
        dtype=np.float32
    )

    add_feature(
        features,
        names,
        np.mean(
            means,
            axis=1
        ),
        f"{prefix}_blockmean_mean"
    )

    add_feature(
        features,
        names,
        np.std(
            means,
            axis=1
        ),
        f"{prefix}_blockmean_std"
    )

    add_feature(
        features,
        names,
        row_slope(means),
        f"{prefix}_blockmean_slope"
    )

    add_feature(
        features,
        names,
        np.mean(
            stds,
            axis=1
        ),
        f"{prefix}_blockstd_mean"
    )


# ============================================================
# BVP / cardiac features
# ============================================================

def vpg_bpm(bvp):
    """
    Approximate heart rate from positive-going VPG
    zero crossings.

    BVP sampling rate = 64 Hz.
    10-second window -> multiply beat count by 6.
    """

    vpg = np.diff(
        bvp,
        axis=1
    )

    crossings = (
        (vpg[:, :-1] <= 0)
        &
        (vpg[:, 1:] > 0)
    )

    beat_count = np.sum(
        crossings,
        axis=1
    )

    return (
        beat_count * 6.0
    ).astype(np.float32)


def dominant_cardiac_period(bvp):
    """
    Find dominant cardiac period from autocorrelation.

    Candidate lags correspond approximately to
    40--180 BPM for a 64 Hz BVP signal.
    """

    lags = np.asarray(
        [
            16,
            20,
            24,
            28,
            32,
            38,
            44,
            50,
            58,
            66,
            76,
            86,
            96,
        ],
        dtype=np.int32
    )

    mean = np.mean(
        bvp,
        axis=1,
        keepdims=True,
        dtype=np.float32
    )

    centered = bvp - mean

    denominator = (
        np.sum(
            centered * centered,
            axis=1
        )
        + 1e-10
    )

    autocorrelations = []

    for lag in lags:

        numerator = np.sum(
            centered[:, :-lag]
            * centered[:, lag:],
            axis=1
        )

        autocorrelations.append(
            numerator / denominator
        )

    autocorrelations = np.column_stack(
        autocorrelations
    )

    best_idx = np.argmax(
        autocorrelations,
        axis=1
    )

    best_lag = lags[
        best_idx
    ]

    best_autocorr = autocorrelations[
        np.arange(bvp.shape[0]),
        best_idx
    ]

    bpm = (
        3840.0 / best_lag
    )

    return (
        bpm.astype(np.float32),
        best_autocorr.astype(np.float32)
    )


def add_vpg_features(
    features,
    names,
    bvp
):
    vpg = np.diff(
        bvp,
        axis=1
    )

    apg = np.diff(
        vpg,
        axis=1
    )

    # Existing derivative features.
    add_basic_features(
        features,
        names,
        vpg,
        "vpg"
    )

    add_basic_features(
        features,
        names,
        apg,
        "apg"
    )

    # Simple BPM estimate.
    add_feature(
        features,
        names,
        vpg_bpm(bvp),
        "vpg_estimated_bpm"
    )

    # Dominant cardiac period.
    dominant_bpm, dominant_ac = (
        dominant_cardiac_period(
            bvp
        )
    )

    add_feature(
        features,
        names,
        dominant_bpm,
        "bvp_dominant_bpm"
    )

    add_feature(
        features,
        names,
        dominant_ac,
        "bvp_dominant_autocorr"
    )

    # Rise/fall energy asymmetry.
    positive_energy = np.mean(
        np.maximum(vpg, 0.0) ** 2,
        axis=1
    )

    negative_energy = (
        np.mean(
            np.minimum(vpg, 0.0) ** 2,
            axis=1
        )
        + 1e-10
    )

    add_feature(
        features,
        names,
        positive_energy / negative_energy,
        "bvp_rise_fall_energy_ratio"
    )

    # Positive VPG strength.
    add_feature(
        features,
        names,
        np.mean(
            np.maximum(vpg, 0.0),
            axis=1
        ),
        "bvp_rise_strength"
    )

    # Negative VPG strength.
    add_feature(
        features,
        names,
        np.mean(
            np.abs(
                np.minimum(vpg, 0.0)
            ),
            axis=1
        ),
        "bvp_fall_strength"
    )


# ============================================================
# Temporal-position features
# ============================================================

def add_temporal_position_features(
    features,
    names,
    x,
    prefix,
    n_blocks=4
):
    """
    Divide the signal into four temporal regions and
    calculate mean/std/slope for each region.
    """

    n_samples = x.shape[1]

    if n_samples % n_blocks != 0:
        raise ValueError(
            f"{prefix}: cannot split "
            f"{n_samples} samples into "
            f"{n_blocks} equal blocks."
        )

    block_size = (
        n_samples // n_blocks
    )

    for k in range(n_blocks):

        start = k * block_size
        end = start + block_size

        block = x[
            :,
            start:end
        ]

        suffix = (
            f"{prefix}_t{k + 1}"
        )

        add_feature(
            features,
            names,
            row_mean(block),
            f"{suffix}_mean"
        )

        add_feature(
            features,
            names,
            row_std(block),
            f"{suffix}_std"
        )

        add_feature(
            features,
            names,
            row_slope(block),
            f"{suffix}_slope"
        )


# ============================================================
# Complete feature extraction
# ============================================================

def extract_features(
    X_raw,
    feature_columns
):
    """
    Create the final 176 manually engineered features.
    """

    features = []
    names = []

    # --------------------------------------------------------
    # Identify columns.
    # --------------------------------------------------------

    acc_x_idx = [
        i
        for i, c in enumerate(feature_columns)
        if c.startswith("acc_x_")
    ]

    acc_y_idx = [
        i
        for i, c in enumerate(feature_columns)
        if c.startswith("acc_y_")
    ]

    acc_z_idx = [
        i
        for i, c in enumerate(feature_columns)
        if c.startswith("acc_z_")
    ]

    bvp_idx = [
        i
        for i, c in enumerate(feature_columns)
        if c.startswith("bvp_")
    ]

    eda_idx = [
        i
        for i, c in enumerate(feature_columns)
        if c.startswith("eda_")
    ]

    # --------------------------------------------------------
    # Validate raw dataset structure.
    # --------------------------------------------------------

    if len(acc_x_idx) != 320:
        raise ValueError(
            f"Expected 320 acc_x values, "
            f"got {len(acc_x_idx)}"
        )

    if len(acc_y_idx) != 320:
        raise ValueError(
            f"Expected 320 acc_y values, "
            f"got {len(acc_y_idx)}"
        )

    if len(acc_z_idx) != 320:
        raise ValueError(
            f"Expected 320 acc_z values, "
            f"got {len(acc_z_idx)}"
        )

    if len(bvp_idx) != 640:
        raise ValueError(
            f"Expected 640 BVP values, "
            f"got {len(bvp_idx)}"
        )

    if len(eda_idx) != 40:
        raise ValueError(
            f"Expected 40 EDA values, "
            f"got {len(eda_idx)}"
        )

    acc_x = X_raw[:, acc_x_idx]
    acc_y = X_raw[:, acc_y_idx]
    acc_z = X_raw[:, acc_z_idx]

    bvp = X_raw[:, bvp_idx]
    eda = X_raw[:, eda_idx]

    # ========================================================
    # 1. BVP
    # ========================================================

    cardiac_lags = [
        16,
        20,
        24,
        28,
        32,
        38,
        44,
        50,
        58,
        66,
        76,
        86,
        96,
    ]

    add_deep_features(
        features,
        names,
        bvp,
        "bvp",
        cardiac_lags
    )

    add_block_summary(
        features,
        names,
        bvp,
        64,
        "bvp"
    )

    add_vpg_features(
        features,
        names,
        bvp
    )

    add_temporal_position_features(
        features,
        names,
        bvp,
        "bvp"
    )

    # ========================================================
    # 2. ACCELEROMETER
    # ========================================================

    add_basic_features(
        features,
        names,
        acc_x,
        "acc_x"
    )

    add_basic_features(
        features,
        names,
        acc_y,
        "acc_y"
    )

    add_basic_features(
        features,
        names,
        acc_z,
        "acc_z"
    )

    acc_sq = (
        acc_x * acc_x
        + acc_y * acc_y
        + acc_z * acc_z
    )

    acc_sq_mean = np.mean(
        acc_sq,
        axis=1,
        keepdims=True
    )

    acc_sq_norm = (
        acc_sq
        /
        np.where(
            acc_sq_mean < 1e-7,
            1.0,
            acc_sq_mean
        )
    )

    add_deep_features(
        features,
        names,
        acc_sq,
        "acc_sq",
        [
            1,
            2,
            4,
            8,
            16,
        ]
    )

    add_basic_features(
        features,
        names,
        acc_sq_norm,
        "acc_sq_norm"
    )

    add_block_summary(
        features,
        names,
        acc_sq,
        32,
        "acc_sq"
    )

    jerk = np.diff(
        acc_sq,
        axis=1
    )

    add_basic_features(
        features,
        names,
        jerk,
        "jerk"
    )

    # Movement burstiness.
    acc_sq_sorted = np.sort(acc_sq, axis=1)
    idx_10 = int(0.10 * (acc_sq.shape[1] - 1))
    idx_90 = int(0.90 * (acc_sq.shape[1] - 1))
    burstiness = acc_sq_sorted[:, idx_90] - acc_sq_sorted[:, idx_10]

    add_feature(features, names, burstiness, "acc_sq_burstiness")

    # Rapid changes in movement.
    second_diff = np.diff(
        acc_sq,
        n=2,
        axis=1
    )

    add_feature(
        features,
        names,
        np.mean(
            np.abs(second_diff),
            axis=1
        ),
        "acc_sq_second_diff_abs_mean"
    )

    add_feature(
        features,
        names,
        np.std(
            second_diff,
            axis=1
        ),
        "acc_sq_second_diff_std"
    )

    add_temporal_position_features(
        features,
        names,
        acc_sq,
        "acc_sq"
    )

    # ========================================================
    # 3. EDA
    # ========================================================

    add_deep_features(
        features,
        names,
        eda,
        "eda",
        [
            1,
            2,
            4,
        ]
    )

    add_block_summary(
        features,
        names,
        eda,
        4,
        "eda"
    )

    add_temporal_position_features(
        features,
        names,
        eda,
        "eda"
    )

    # ========================================================
    # 4. Physiologically motivated interactions
    # ========================================================

    bvp_bpm = vpg_bpm(
        bvp
    )

    acc_rms = row_rms(
        acc_sq
    )

    eda_mean = row_mean(
        eda
    )

    bvp_std = row_std(
        bvp
    )

    acc_std = row_std(
        acc_sq
    )

    add_feature(
        features,
        names,
        bvp_bpm * acc_rms,
        "interaction_bpm_motion"
    )

    add_feature(
        features,
        names,
        bvp_std * acc_std,
        "interaction_bvp_motion_variability"
    )

    add_feature(
        features,
        names,
        bvp_bpm * eda_mean,
        "interaction_bpm_eda"
    )

    add_feature(
        features,
        names,
        acc_rms * eda_mean,
        "interaction_motion_eda"
    )

    # --------------------------------------------------------
    # Final matrix.
    # --------------------------------------------------------

    Z = np.column_stack(
        features
    ).astype(
        np.float32
    )

    if Z.shape[1] != EXPECTED_ENGINEERED_FEATURES:
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_ENGINEERED_FEATURES} "
            f"engineered features, "
            f"got {Z.shape[1]}"
        )

    if len(names) != Z.shape[1]:
        raise RuntimeError(
            "Feature names and matrix columns "
            "do not match."
        )

    if not np.all(
        np.isfinite(Z)
    ):
        raise ValueError(
            "Feature matrix contains NaN or Inf."
        )

    return Z, names


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage:\n"
            "python3 part_c.py "
            "train.csv test.csv predictions.txt"
        )

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    predictions_path = sys.argv[3]

    # ========================================================
    # TRAINING DATA
    # ========================================================

    print("Loading training CSV...")

    train_df = pd.read_csv(
        train_path
    )

    if "hr" not in train_df.columns:
        raise ValueError(
            "Target column 'hr' not found "
            "in training data."
        )

    feature_columns = [
        c
        for c in train_df.columns
        if c != "hr"
    ]

    if len(feature_columns) != EXPECTED_RAW_FEATURES:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_RAW_FEATURES} raw features, "
            f"got {len(feature_columns)}"
        )

    y_train = train_df[
        "hr"
    ].to_numpy(
        dtype=np.float64
    )

    X_train_raw = train_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    del train_df

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    print(
        "Creating comprehensive "
        "domain-engineered features..."
    )

    Z_train, feature_names = extract_features(
        X_train_raw,
        feature_columns
    )

    del X_train_raw

    print(
        f"Feature matrix: "
        f"{Z_train.shape}"
    )

    print(
        f"Number of engineered features: "
        f"{len(feature_names)}"
    )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print(
        "Fitting StandardScaler "
        "on training data..."
    )

    scaler = StandardScaler()

    Z_train_scaled = scaler.fit_transform(
        Z_train
    )

    del Z_train

    # ========================================================
    # HUBER REGRESSION
    # ========================================================

    print()
    print(
        "Fitting final HuberRegressor..."
    )

    model = HuberRegressor(
        alpha=HUBER_ALPHA,
        epsilon=HUBER_EPSILON,
        max_iter=HUBER_MAX_ITER
    )

    model.fit(
        Z_train_scaled,
        y_train
    )

    print(
        "Huber model fitted successfully."
    )

    print(
        f"alpha   = {HUBER_ALPHA}"
    )

    print(
        f"epsilon = {HUBER_EPSILON}"
    )

    print(
        f"iterations used = {model.n_iter_}"
    )

    del Z_train_scaled
    del y_train

    # ========================================================
    # TEST DATA
    # ========================================================

    print()
    print(
        "Loading test CSV..."
    )

    test_df = pd.read_csv(
        test_path
    )

    X_test_raw = test_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    del test_df

    # ========================================================
    # TEST FEATURE ENGINEERING
    # ========================================================

    print(
        "Creating test engineered features..."
    )

    Z_test, test_feature_names = extract_features(
        X_test_raw,
        feature_columns
    )

    del X_test_raw

    if test_feature_names != feature_names:
        raise ValueError(
            "Training and test feature "
            "orders do not match."
        )

    # ========================================================
    # SCALE TEST FEATURES
    # ========================================================

    Z_test_scaled = scaler.transform(
        Z_test
    )

    del Z_test

    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "Generating predictions..."
    )

    predictions = model.predict(
        Z_test_scaled
    )

    if not np.all(
        np.isfinite(predictions)
    ):
        raise ValueError(
            "Predictions contain NaN or Inf."
        )

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    np.savetxt(
        predictions_path,
        predictions,
        fmt="%.10f"
    )

    print(
        f"Successfully saved "
        f"{len(predictions)} predictions "
        f"to {predictions_path}"
    )


if __name__ == "__main__":
    main()