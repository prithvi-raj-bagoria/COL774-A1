import sys
import numpy as np
import pandas as pd


def calculate_metrics(y_true, y_pred):
    """Calculate MAE, MSE, NMAE and NMSE."""

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    y_mean = np.mean(y_true)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    nmae = (
        np.sum(np.abs(y_true - y_pred))
        /
        np.sum(np.abs(y_true - y_mean))
    )

    nmse = (
        np.sum((y_true - y_pred) ** 2)
        /
        np.sum((y_true - y_mean) ** 2)
    )

    return mae, mse, nmae, nmse


def main():
    """
    Usage:

        python src/evaluate.py test.csv predictions.txt model_name

    Or evaluate multiple prediction files:

        python src/evaluate.py test.csv pred_a.txt pred_b.txt
    """

    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage:\n"
            "python src/evaluate.py test.csv predictions.txt "
            "[predictions2.txt ...]"
        )

    test_path = sys.argv[1]
    prediction_files = sys.argv[2:]

    test = pd.read_csv(test_path)

    if "hr" not in test.columns:
        raise ValueError(
            "Test CSV must contain the 'hr' column "
            "for local evaluation."
        )

    y_true = test["hr"].to_numpy(
        dtype=np.float64
    )

    print()

    print(
        f"{'Prediction file':<32}"
        f"{'MAE':>14}"
        f"{'MSE':>14}"
        f"{'NMAE':>14}"
        f"{'NMSE':>14}"
    )

    print("-" * 88)

    for prediction_file in prediction_files:

        y_pred = np.loadtxt(
            prediction_file,
            dtype=np.float64
        )

        y_pred = np.asarray(
            y_pred
        ).reshape(-1)

        mae, mse, nmae, nmse = calculate_metrics(
            y_true,
            y_pred
        )

        print(
            f"{prediction_file:<32}"
            f"{mae:>14.6f}"
            f"{mse:>14.6f}"
            f"{nmae:>14.6f}"
            f"{nmse:>14.6f}"
        )


if __name__ == "__main__":
    main()