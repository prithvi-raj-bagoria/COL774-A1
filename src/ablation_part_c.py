import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

from part_c import (
    ALPHAS,
    extract_features_from_array,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_PATH = "data/e4_hr_train_downsampled.csv"

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20


# ============================================================
# Identify modality from feature name
# ============================================================

def feature_group(name):

    # Cross-modality feature MUST be checked first.
    if name == "bvp_acc_cross_corr":
        return "INTERACTION"

    if (
        name.startswith("bvp_")
        or name.startswith("vpg_")
        or name.startswith("apg_")
        or name == "vpg_estimated_bpm"
    ):
        return "BVP"

    if (
        name.startswith("acc_")
        or name.startswith("jerk_")
    ):
        return "ACC"

    if name.startswith("eda_"):
        return "EDA"

    return "OTHER"

    # BVP-only features
    if (
        name.startswith("bvp_")
        or name.startswith("vpg_")
        or name.startswith("apg_")
        or name == "vpg_estimated_bpm"
    ):
        return "BVP"

    # Accelerometer features
    if (
        name.startswith("acc_")
        or name.startswith("jerk_")
    ):
        return "ACC"

    # EDA features
    if name.startswith("eda_"):
        return "EDA"

    # Cross-modality feature
    if name == "bvp_acc_cross_corr":
        return "INTERACTION"

    return "OTHER"


# ============================================================
# Metrics
# ============================================================

def nmae(y_true, y_pred):
    mean_y = np.mean(y_true)

    return (
        np.sum(np.abs(y_true - y_pred))
        /
        np.sum(np.abs(y_true - mean_y))
    )


def nmse(y_true, y_pred):
    mean_y = np.mean(y_true)

    return (
        np.sum((y_true - y_pred) ** 2)
        /
        np.sum((y_true - mean_y) ** 2)
    )


# ============================================================
# Train one modality combination
# ============================================================

def run_model(
    Z_train,
    y_train,
    Z_val,
    y_val,
    feature_names,
    allowed_groups,
):
    # --------------------------------------------------------
    # Select columns
    # --------------------------------------------------------

    selected = []

    for i, name in enumerate(feature_names):

        group = feature_group(name)

        if group in allowed_groups:
            selected.append(i)

    if not selected:
        raise ValueError(
            f"No features found for groups: {allowed_groups}"
        )

    Xtr = Z_train[:, selected]
    Xva = Z_val[:, selected]

    # --------------------------------------------------------
    # Same modeling pipeline as Part (c):
    #
    # StandardScaler + RidgeCV
    # --------------------------------------------------------

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS)
    )

    model.fit(
        Xtr,
        y_train
    )

    pred = model.predict(
        Xva
    )

    alpha = model.named_steps[
        "ridgecv"
    ].alpha_

    return {
        "features": len(selected),
        "alpha": alpha,
        "nmae": nmae(y_val, pred),
        "nmse": nmse(y_val, pred),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("Loading training data...")

    df = pd.read_csv(
        TRAIN_PATH
    )

    feature_columns = [
        col for col in df.columns
        if col != "hr"
    ]

    y = df["hr"].to_numpy(
        dtype=np.float64
    )

    X_raw = df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    del df

    # --------------------------------------------------------
    # SAME feature engineering as Part (c)
    # --------------------------------------------------------

    print("Extracting the 129 Part (c) features...")

    Z, feature_names = extract_features_from_array(
        X_raw,
        feature_columns
    )

    del X_raw

    print(
        f"Total features: {Z.shape[1]}"
    )

    # --------------------------------------------------------
    # Print feature-group sizes
    # --------------------------------------------------------

    group_counts = {
        "BVP": 0,
        "ACC": 0,
        "EDA": 0,
        "INTERACTION": 0,
        "OTHER": 0,
    }

    for name in feature_names:
        group_counts[
            feature_group(name)
        ] += 1

    print("\nFeature groups:")

    for group, count in group_counts.items():
        print(
            f"  {group:<15} {count}"
        )

    # --------------------------------------------------------
    # FIXED validation split
    #
    # Every ablation uses EXACTLY the same examples.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = np.arange(
        len(y)
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

    print(
        f"\nTraining examples : {len(train_idx)}"
    )

    print(
        f"Validation examples: {len(val_idx)}"
    )

    # --------------------------------------------------------
    # Ablation configurations
    # --------------------------------------------------------

    experiments = [
        (
            "BVP + ACC",
            {"BVP", "ACC"}
        ),
        (
            "BVP + ACC + EDA",
            {"BVP", "ACC", "EDA"}
        ),
        (
            "BVP + ACC + Interaction",
            {"BVP", "ACC", "INTERACTION"}
        ),
        (
            "ALL",
            {"BVP", "ACC", "EDA", "INTERACTION"}
        ),
    ]

    results = []

    print()
    print("=" * 100)
    print("PART (C) MODALITY ABLATION")
    print("=" * 100)

    print(
        f"{'Model':<20}"
        f"{'Features':>12}"
        f"{'Alpha':>14}"
        f"{'NMAE':>14}"
        f"{'NMSE':>14}"
    )

    print("-" * 100)

    for name, groups in experiments:

        result = run_model(
            Z_train,
            y_train,
            Z_val,
            y_val,
            feature_names,
            groups,
        )

        results.append(
            (name, result)
        )

        print(
            f"{name:<20}"
            f"{result['features']:>12}"
            f"{result['alpha']:>14.6f}"
            f"{result['nmae']:>14.6f}"
            f"{result['nmse']:>14.6f}"
        )

    # --------------------------------------------------------
    # Best by validation NMSE
    # --------------------------------------------------------

    best_nmse = min(
        results,
        key=lambda x: x[1]["nmse"]
    )

    best_nmae = min(
        results,
        key=lambda x: x[1]["nmae"]
    )

    print()
    print("=" * 100)
    print("BEST MODELS")
    print("=" * 100)

    print(
        f"Best validation NMSE: "
        f"{best_nmse[0]} "
        f"({best_nmse[1]['nmse']:.6f})"
    )

    print(
        f"Best validation NMAE: "
        f"{best_nmae[0]} "
        f"({best_nmae[1]['nmae']:.6f})"
    )


if __name__ == "__main__":
    main()
