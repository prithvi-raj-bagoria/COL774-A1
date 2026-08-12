import sys
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python3 part_a.py "
            "train.csv test.csv predictions.txt weights.txt"
        )

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    predictions_path = sys.argv[3]
    weights_path = sys.argv[4]

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # ---------------------------------------------------------
    # Actual supplied files:
    # 1640 feature columns + hr
    # ---------------------------------------------------------
    feature_columns = train.columns[:-1]

    # Use exactly the 1640 feature columns, in their original order.
    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train["hr"].to_numpy(dtype=float)

    X_test = test[feature_columns].to_numpy(dtype=float)

    # Sanity checks
    if X_train.shape[1] != 1640:
        raise ValueError(
            f"Expected 1640 features, found {X_train.shape[1]}"
        )

    if X_test.shape[1] != 1640:
        raise ValueError(
            f"Expected 1640 test features, found {X_test.shape[1]}"
        )

    # ---------------------------------------------------------
    # Add intercept column:
    #
    # X_aug = [1, x1, ..., x1640]
    # ---------------------------------------------------------
    X_train_aug = np.hstack(
        (np.ones((X_train.shape[0], 1)), X_train)
    )

    X_test_aug = np.hstack(
        (np.ones((X_test.shape[0], 1)), X_test)
    )

    # ---------------------------------------------------------
    # Closed-form OLS:
    #
    # W = (X^T X)^(-1) X^T Y
    # ---------------------------------------------------------
    XtX = X_train_aug.T @ X_train_aug
    XtY = X_train_aug.T @ y_train

    W = np.linalg.inv(XtX) @ XtY

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    predictions = X_test_aug @ W

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    np.savetxt(
        predictions_path,
        predictions,
        fmt="%.10f"
    )

    np.savetxt(
        weights_path,
        W,
        fmt="%.10f"
    )


if __name__ == "__main__":
    main()