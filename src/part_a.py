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

    # The training file contains 'hr' target column after all 1640 feature columns
    # Extract feature column names preserving exact order
    feature_columns = [col for col in train.columns if col != "hr"]

    X_train = train[feature_columns].to_numpy(dtype=np.float64)
    y_train = train["hr"].to_numpy(dtype=np.float64)

    X_test = test[feature_columns].to_numpy(dtype=np.float64)

    # Sanity checks
    if X_train.shape[1] != 1640:
        raise ValueError(
            f"Expected 1640 features, found {X_train.shape[1]}"
        )

    if X_test.shape[1] != 1640:
        raise ValueError(
            f"Expected 1640 test features, found {X_test.shape[1]}"
        )

    # Construct augmented matrix X_aug = [1, x1, ..., x1640]
    ones_train = np.ones((X_train.shape[0], 1), dtype=np.float64)
    X_train_aug = np.hstack((ones_train, X_train))

    ones_test = np.ones((X_test.shape[0], 1), dtype=np.float64)
    X_test_aug = np.hstack((ones_test, X_test))

    # Closed-form OLS: W = (X^T X)^(-1) X^T Y
    # Note: Assignment explicitly mandates using np.linalg.inv
    XtX = X_train_aug.T @ X_train_aug
    XtY = X_train_aug.T @ y_train

    # Explicit inverse per prompt instructions
    XtX_inv = np.linalg.inv(XtX)
    W = XtX_inv @ XtY

    # Compute predictions
    predictions = X_test_aug @ W

    # Save output files preserving original order
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    np.savetxt(weights_path, W, fmt="%.10f")


if __name__ == "__main__":
    main()