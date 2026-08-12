import sys
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20

# A small set because the feature matrix is low-dimensional
# after feature engineering.
LAMBDA_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


# ============================================================
# Basic feature helpers
# ============================================================

def row_mean(x):
    return np.mean(x, axis=1)


def row_std(x):
    return np.std(x, axis=1)


def row_min(x):
    return np.min(x, axis=1)


def row_max(x):
    return np.max(x, axis=1)


def row_range(x):
    return np.max(x, axis=1) - np.min(x, axis=1)


def row_median(x):
    return np.median(x, axis=1)


def row_q25(x):
    return np.percentile(x, 25, axis=1)


def row_q75(x):
    return np.percentile(x, 75, axis=1)


def row_rms(x):
    return np.sqrt(np.mean(x * x, axis=1))


def row_energy(x):
    return np.mean(x * x, axis=1)


def row_abs_diff_mean(x):
    return np.mean(np.abs(np.diff(x, axis=1)), axis=1)


def row_diff_std(x):
    return np.std(np.diff(x, axis=1), axis=1)


def row_slope(x):
    """
    Least-squares slope of each row against equally spaced
    time samples.
    """
    n = x.shape[1]

    t = np.arange(n, dtype=np.float64)
    t = t - np.mean(t)

    denominator = np.sum(t * t)

    return (x @ t) / denominator


def row_last_minus_first(x):
    return x[:, -1] - x[:, 0]


def row_lag1_autocorrelation(x):
    """
    Lag-1 autocorrelation.
    """
    mean = np.mean(x, axis=1, keepdims=True)

    centered = x - mean

    numerator = np.sum(
        centered[:, :-1] * centered[:, 1:],
        axis=1
    )

    denominator = np.sum(
        centered * centered,
        axis=1
    )

    denominator = np.maximum(denominator, 1e-12)

    return numerator / denominator


# ============================================================
# Feature extraction
# ============================================================

def add_feature(features, names, values, name):
    """
    Append one feature column.
    """
    features.append(values)
    names.append(name)


def add_basic_features(features, names, x, prefix):
    """
    Add general statistical / temporal features to a signal.
    """

    add_feature(
        features, names,
        row_mean(x),
        f"{prefix}_mean"
    )

    add_feature(
        features, names,
        row_std(x),
        f"{prefix}_std"
    )

    add_feature(
        features, names,
        row_min(x),
        f"{prefix}_min"
    )

    add_feature(
        features, names,
        row_max(x),
        f"{prefix}_max"
    )

    add_feature(
        features, names,
        row_range(x),
        f"{prefix}_range"
    )

    add_feature(
        features, names,
        row_median(x),
        f"{prefix}_median"
    )

    add_feature(
        features, names,
        row_q25(x),
        f"{prefix}_q25"
    )

    add_feature(
        features, names,
        row_q75(x),
        f"{prefix}_q75"
    )

    add_feature(
        features, names,
        row_rms(x),
        f"{prefix}_rms"
    )

    add_feature(
        features, names,
        row_energy(x),
        f"{prefix}_energy"
    )

    add_feature(
        features, names,
        row_abs_diff_mean(x),
        f"{prefix}_mean_abs_diff"
    )

    add_feature(
        features, names,
        row_diff_std(x),
        f"{prefix}_diff_std"
    )

    add_feature(
        features, names,
        row_slope(x),
        f"{prefix}_slope"
    )

    add_feature(
        features, names,
        row_last_minus_first(x),
        f"{prefix}_last_minus_first"
    )

    add_feature(
        features, names,
        row_lag1_autocorrelation(x),
        f"{prefix}_lag1_autocorr"
    )
        # Additional temporal features
    if prefix == "bvp":
        add_feature(
            features,
            names,
            row_lag1_autocorrelation(x),
            f"{prefix}_lag1_autocorr_extra"
        )

        d2 = np.diff(x, n=2, axis=1)

        add_feature(
            features,
            names,
            np.mean(np.abs(d2), axis=1),
            f"{prefix}_second_diff_abs_mean_extra"
        )

        add_feature(
            features,
            names,
            np.std(d2, axis=1),
            f"{prefix}_second_diff_std_extra"
        )

        add_feature(
            features,
            names,
            np.percentile(x, 90, axis=1)
            - np.percentile(x, 10, axis=1),
            f"{prefix}_p90_p10_extra"
        )

def block_features(
    x,
    block_size,
    prefix,
    features,
    names
):
    """
    Divide a signal into equal temporal blocks.

    For each block we calculate:
        mean
        std
        range

    Then summarize those block-level quantities.
    """

    n_samples = x.shape[1]

    if n_samples % block_size != 0:
        raise ValueError(
            f"{prefix}: signal length {n_samples} "
            f"is not divisible by block size {block_size}"
        )

    n_blocks = n_samples // block_size

    blocks = x.reshape(
        x.shape[0],
        n_blocks,
        block_size
    )

    means = np.mean(blocks, axis=2)
    stds = np.std(blocks, axis=2)
    ranges = np.max(blocks, axis=2) - np.min(
        blocks, axis=2
    )

    # Average across temporal blocks
    add_feature(
        features, names,
        np.mean(means, axis=1),
        f"{prefix}_blockmean_mean"
    )

    add_feature(
        features, names,
        np.std(means, axis=1),
        f"{prefix}_blockmean_std"
    )

    add_feature(
        features, names,
        np.min(means, axis=1),
        f"{prefix}_blockmean_min"
    )

    add_feature(
        features, names,
        np.max(means, axis=1),
        f"{prefix}_blockmean_max"
    )

    add_feature(
        features, names,
        means[:, -1] - means[:, 0],
        f"{prefix}_blockmean_change"
    )

    # Trend across temporal blocks
    add_feature(
        features, names,
        row_slope(means),
        f"{prefix}_blockmean_slope"
    )

    # Block standard deviations
    add_feature(
        features, names,
        np.mean(stds, axis=1),
        f"{prefix}_blockstd_mean"
    )

    add_feature(
        features, names,
        np.std(stds, axis=1),
        f"{prefix}_blockstd_std"
    )

    # Block ranges
    add_feature(
        features, names,
        np.mean(ranges, axis=1),
        f"{prefix}_blockrange_mean"
    )

    add_feature(
        features, names,
        np.max(ranges, axis=1),
        f"{prefix}_blockrange_max"
    )


def extract_features(
    df,
    feature_columns
):
    """
    Convert the 1640 raw physiological values into
    manually engineered features.

    No automatic feature-extraction library is used.
    """

    X_raw = df[feature_columns].to_numpy(
        dtype=np.float64
    )

    n = X_raw.shape[0]

    features = []
    names = []

    # --------------------------------------------------------
    # Locate the actual columns using the professor's naming
    # convention.
    # --------------------------------------------------------

    acc_x_idx = [
        i for i, c in enumerate(feature_columns)
        if c.startswith("acc_x_")
    ]

    acc_y_idx = [
        i for i, c in enumerate(feature_columns)
        if c.startswith("acc_y_")
    ]

    acc_z_idx = [
        i for i, c in enumerate(feature_columns)
        if c.startswith("acc_z_")
    ]

    bvp_idx = [
        i for i, c in enumerate(feature_columns)
        if c.startswith("bvp_")
    ]

    eda_idx = [
        i for i, c in enumerate(feature_columns)
        if c.startswith("eda_")
    ]

    # --------------------------------------------------------
    # Sanity checks against the assignment structure.
    # --------------------------------------------------------

    if len(acc_x_idx) != 320:
        raise ValueError(
            f"Expected 320 acc_x samples, got {len(acc_x_idx)}"
        )

    if len(acc_y_idx) != 320:
        raise ValueError(
            f"Expected 320 acc_y samples, got {len(acc_y_idx)}"
        )

    if len(acc_z_idx) != 320:
        raise ValueError(
            f"Expected 320 acc_z samples, got {len(acc_z_idx)}"
        )

    if len(bvp_idx) != 640:
        raise ValueError(
            f"Expected 640 BVP samples, got {len(bvp_idx)}"
        )

    if len(eda_idx) != 40:
        raise ValueError(
            f"Expected 40 EDA samples, got {len(eda_idx)}"
        )

    # --------------------------------------------------------
    # Extract modalities.
    # --------------------------------------------------------

    acc_x = X_raw[:, acc_x_idx]
    acc_y = X_raw[:, acc_y_idx]
    acc_z = X_raw[:, acc_z_idx]

    bvp = X_raw[:, bvp_idx]
    eda = X_raw[:, eda_idx]

    # --------------------------------------------------------
    # BVP features
    #
    # 640 samples = 10 seconds at 64 Hz.
    # --------------------------------------------------------

    add_basic_features(
        features,
        names,
        bvp,
        "bvp"
    )

    block_features(
        bvp,
        64,
        "bvp",
        features,
        names
    )

    # --------------------------------------------------------
    # Accelerometer axis features
    #
    # Each axis has 320 samples.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Squared acceleration magnitude.
    #
    # Assignment report explicitly defines:
    #
    # asq(t) = ax(t)^2 + ay(t)^2 + az(t)^2
    # --------------------------------------------------------

    acc_sq = (
        acc_x * acc_x
        + acc_y * acc_y
        + acc_z * acc_z
    )

    add_basic_features(
        features,
        names,
        acc_sq,
        "acc_sq"
    )

    block_features(
        acc_sq,
        32,
        "acc_sq",
        features,
        names
    )

    # --------------------------------------------------------
    # Cross-axis interactions.
    # --------------------------------------------------------

    add_feature(
        features,
        names,
        np.mean(acc_x * acc_y, axis=1),
        "acc_xy_mean"
    )

    add_feature(
        features,
        names,
        np.mean(acc_x * acc_z, axis=1),
        "acc_xz_mean"
    )

    add_feature(
        features,
        names,
        np.mean(acc_y * acc_z, axis=1),
        "acc_yz_mean"
    )

    # --------------------------------------------------------
    # EDA
    #
    # 40 samples = 10 seconds at 4 Hz.
    # --------------------------------------------------------

    add_basic_features(
        features,
        names,
        eda,
        "eda"
    )

    block_features(
        eda,
        4,
        "eda",
        features,
        names
    )

    # --------------------------------------------------------
    # Combine.
    # --------------------------------------------------------

    Z = np.column_stack(features)

    if Z.shape[0] != n:
        raise RuntimeError("Feature row count changed.")

    if not np.all(np.isfinite(Z)):
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return Z, names


# ============================================================
# Ridge regression
# ============================================================

def fit_ridge_standardized(
    X,
    y,
    lam
):
    """
    Standardize features and fit ridge regression.

    The intercept is not regularized.

    Returns:
        intercept
        coefficients
        mean
        std
    """

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Avoid division by zero for constant features.
    std = np.where(std < 1e-12, 1.0, std)

    Xs = (X - mean) / std

    # Add bias column.
    Xa = np.column_stack(
        (
            np.ones(Xs.shape[0]),
            Xs
        )
    )

    # Regularization matrix.
    R = np.eye(
        Xa.shape[1],
        dtype=np.float64
    )

    R[0, 0] = 0.0

    A = Xa.T @ Xa
    b = Xa.T @ y

    W = np.linalg.inv(
        A + lam * R
    ) @ b

    return W[0], W[1:], mean, std


def predict_standardized(
    X,
    intercept,
    coef,
    mean,
    std
):
    Xs = (X - mean) / std

    return intercept + Xs @ coef


# ============================================================
# Metrics
# ============================================================

def nmae(y_true, y_pred):
    mean_target = np.mean(y_true)

    return (
        np.sum(np.abs(y_true - y_pred))
        /
        np.sum(np.abs(y_true - mean_target))
    )


def nmse(y_true, y_pred):
    mean_target = np.mean(y_true)

    return (
        np.sum((y_true - y_pred) ** 2)
        /
        np.sum((y_true - mean_target) ** 2)
    )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: python3 part_c.py "
            "train.csv test.csv predictions.txt"
        )

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    predictions_path = sys.argv[3]

    # --------------------------------------------------------
    # Read train data.
    # --------------------------------------------------------

    train_df = pd.read_csv(train_path)

    if train_df.columns[-1] != "hr":
        raise ValueError(
            f"Expected target column 'hr', "
            f"found '{train_df.columns[-1]}'"
        )

    feature_columns = list(
        train_df.columns[:-1]
    )

    if len(feature_columns) != 1640:
        raise ValueError(
            f"Expected 1640 raw features, "
            f"found {len(feature_columns)}"
        )

    y = train_df["hr"].to_numpy(
        dtype=np.float64
    )

    print("Creating engineered training features...")

    Z, feature_names = extract_features(
        train_df,
        feature_columns
    )

    print(
        f"Engineered feature matrix: {Z.shape}"
    )

    print(
        f"Number of engineered features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Deterministic random validation split.
    #
    # This is ONLY used for local model selection.
    # It is not the supplied test set.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = np.arange(
        Z.shape[0]
    )

    rng.shuffle(indices)

    n_val = int(
        VALIDATION_FRACTION * len(indices)
    )

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    Z_train = Z[train_idx]
    y_train = y[train_idx]

    Z_val = Z[val_idx]
    y_val = y[val_idx]

    # --------------------------------------------------------
    # Select ridge lambda.
    # --------------------------------------------------------

    best_lambda = None
    best_nmse = np.inf

    print("\nValidation results:")

    for lam in LAMBDA_VALUES:

        intercept, coef, mean, std = (
            fit_ridge_standardized(
                Z_train,
                y_train,
                lam
            )
        )

        pred = predict_standardized(
            Z_val,
            intercept,
            coef,
            mean,
            std
        )

        score_nmse = nmse(
            y_val,
            pred
        )

        score_nmae = nmae(
            y_val,
            pred
        )

        print(
            f"lambda={lam:8.4f} "
            f"NMAE={score_nmae:.8f} "
            f"NMSE={score_nmse:.8f}"
        )

        if score_nmse < best_nmse:
            best_nmse = score_nmse
            best_lambda = lam

    print(
        f"\nSelected lambda: {best_lambda}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # After selecting lambda, refit using ALL training data.
    # --------------------------------------------------------

    intercept, coef, mean, std = (
        fit_ridge_standardized(
            Z,
            y,
            best_lambda
        )
    )

    # --------------------------------------------------------
    # Load test only now.
    # --------------------------------------------------------

    print("\nCreating engineered test features...")

    test_df = pd.read_csv(
        test_path
    )

    Z_test, test_feature_names = (
        extract_features(
            test_df,
            feature_columns
        )
    )

    if test_feature_names != feature_names:
        raise ValueError(
            "Training and test feature order differ."
        )

    # --------------------------------------------------------
    # Predict.
    # --------------------------------------------------------

    predictions = predict_standardized(
        Z_test,
        intercept,
        coef,
        mean,
        std
    )

    if not np.all(
        np.isfinite(predictions)
    ):
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    # --------------------------------------------------------
    # Required output.
    # --------------------------------------------------------

    np.savetxt(
        predictions_path,
        predictions,
        fmt="%.10f"
    )

    print(
        f"\nWrote {len(predictions)} predictions "
        f"to {predictions_path}"
    )


if __name__ == "__main__":
    main()