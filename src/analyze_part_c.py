import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline

from part_c import (
    ALPHAS,
    extract_features_from_array,
)


TRAIN_PATH = "data/e4_hr_train_downsampled.csv"


def modality(feature_name):
    if (
        feature_name.startswith("bvp")
        or feature_name.startswith("vpg")
        or feature_name.startswith("apg")
        or feature_name.startswith("vpg_estimated")
    ):
        return "BVP"

    if (
        feature_name.startswith("acc")
        or feature_name.startswith("jerk")
    ):
        return "Accelerometer"

    if feature_name.startswith("eda"):
        return "EDA"

    return "Other"


def main():
    # ========================================================
    # Load training data
    # ========================================================

    train_df = pd.read_csv(TRAIN_PATH)

    feature_columns = [
        col for col in train_df.columns
        if col != "hr"
    ]

    y = train_df["hr"].to_numpy(
        dtype=np.float64
    )

    X_raw = train_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    del train_df

    # ========================================================
    # EXACT SAME FEATURE ENGINEERING AS PART C
    # ========================================================

    print("Extracting features...")

    Z, feature_names = extract_features_from_array(
        X_raw,
        feature_columns
    )

    del X_raw

    print(
        f"Feature matrix: {Z.shape}"
    )

    # ========================================================
    # EXACT SAME MODEL AS PART C
    # ========================================================

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(
            alphas=ALPHAS
        )
    )

    print("Fitting RidgeCV...")

    model.fit(
        Z,
        y
    )

    scaler = model.named_steps[
        "standardscaler"
    ]

    ridge = model.named_steps[
        "ridgecv"
    ]

    print(
        f"Selected alpha: {ridge.alpha_}"
    )

    # ========================================================
    # Standardized regression coefficients
    #
    # Because StandardScaler comes before RidgeCV,
    # ridge.coef_ corresponds to standardized features.
    # ========================================================

    coefficients = np.asarray(
        ridge.coef_
    ).reshape(-1)

    abs_coefficients = np.abs(
        coefficients
    )

    # ========================================================
    # Standardized features
    # ========================================================

    Z_standardized = scaler.transform(
        Z
    )

    # ========================================================
    # Actual contribution for every sample:
    #
    # contribution_ij = beta_j * z_ij
    # ========================================================

    contributions = (
        Z_standardized
        * coefficients
    )

    mean_abs_contribution = np.mean(
        np.abs(contributions),
        axis=0
    )

    mean_signed_contribution = np.mean(
        contributions,
        axis=0
    )

    # ========================================================
    # Rank by absolute standardized coefficient
    # REQUIRED BY ASSIGNMENT
    # ========================================================

    coefficient_order = np.argsort(
        abs_coefficients
    )[::-1]

    print()
    print("=" * 100)
    print("TOP 20 FEATURES BY |STANDARDIZED COEFFICIENT|")
    print("=" * 100)

    print(
        f"{'Rank':<6}"
        f"{'Feature':<40}"
        f"{'Modality':<18}"
        f"{'Coefficient':>15}"
        f"{'|Coefficient|':>15}"
    )

    print("-" * 100)

    for rank, idx in enumerate(
        coefficient_order[:20],
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{feature_names[idx]:<40}"
            f"{modality(feature_names[idx]):<18}"
            f"{coefficients[idx]:>15.6f}"
            f"{abs_coefficients[idx]:>15.6f}"
        )

    # ========================================================
    # Rank by mean absolute prediction contribution
    # ========================================================

    contribution_order = np.argsort(
        mean_abs_contribution
    )[::-1]

    print()
    print("=" * 100)
    print("TOP 20 FEATURES BY MEAN |PREDICTION CONTRIBUTION|")
    print("=" * 100)

    print(
        f"{'Rank':<6}"
        f"{'Feature':<40}"
        f"{'Modality':<18}"
        f"{'Mean |Contribution|':>20}"
        f"{'Mean Contribution':>18}"
    )

    print("-" * 100)

    for rank, idx in enumerate(
        contribution_order[:20],
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{feature_names[idx]:<40}"
            f"{modality(feature_names[idx]):<18}"
            f"{mean_abs_contribution[idx]:>20.6f}"
            f"{mean_signed_contribution[idx]:>18.6f}"
        )

    # ========================================================
    # Required Top 5
    # ========================================================

    print()
    print("=" * 100)
    print("TOP 5 FEATURES FOR THE REPORT")
    print("=" * 100)

    for rank, idx in enumerate(
        coefficient_order[:5],
        start=1
    ):
        print(
            f"{rank}. "
            f"{feature_names[idx]} | "
            f"coefficient = {coefficients[idx]:.8f} | "
            f"|coefficient| = {abs_coefficients[idx]:.8f} | "
            f"mean |contribution| = "
            f"{mean_abs_contribution[idx]:.8f}"
        )

    # ========================================================
    # Modality summary
    # ========================================================

    modality_stats = {}

    for idx, name in enumerate(
        feature_names
    ):
        mod = modality(name)

        if mod not in modality_stats:
            modality_stats[mod] = {
                "abs_coef": 0.0,
                "abs_contribution": 0.0,
                "count": 0,
            }

        modality_stats[mod]["abs_coef"] += (
            abs_coefficients[idx]
        )

        modality_stats[mod]["abs_contribution"] += (
            mean_abs_contribution[idx]
        )

        modality_stats[mod]["count"] += 1

    print()
    print("=" * 100)
    print("MODALITY SUMMARY")
    print("=" * 100)

    for mod, stats in sorted(
        modality_stats.items(),
        key=lambda item:
            item[1]["abs_contribution"],
        reverse=True
    ):

        print(
            f"{mod:<20}"
            f"features={stats['count']:<5}"
            f"sum|coef|={stats['abs_coef']:.6f}   "
            f"sum mean|contribution|="
            f"{stats['abs_contribution']:.6f}"
        )


if __name__ == "__main__":
    main()
