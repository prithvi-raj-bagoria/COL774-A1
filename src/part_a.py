import sys
import numpy as np
import pandas as pd

def main():
    if len(sys.argv) != 5:
        sys.exit("Usage: python3 part_a.py train.csv test.csv predictions.txt weights.txt")

    train_path, test_path, predictions_path, weights_path = sys.argv[1:5]

    train = pd.read_csv(train_path)
    feature_columns = [col for col in train.columns if col != "hr"]
    X_train = train[feature_columns].to_numpy(dtype=np.float64)
    y_train = train["hr"].to_numpy(dtype=np.float64)
    del train

    ones_train = np.ones((X_train.shape[0], 1), dtype=np.float64)
    X_train_aug = np.hstack((ones_train, X_train))
    del X_train, ones_train

    XtX = X_train_aug.T @ X_train_aug
    XtY = X_train_aug.T @ y_train
    del X_train_aug

    W = np.linalg.inv(XtX) @ XtY
    del XtX, XtY

    test = pd.read_csv(test_path)
    X_test = test[feature_columns].to_numpy(dtype=np.float64)
    del test

    ones_test = np.ones((X_test.shape[0], 1), dtype=np.float64)
    X_test_aug = np.hstack((ones_test, X_test))
    del X_test, ones_test

    predictions = X_test_aug @ W
    del X_test_aug

    np.savetxt(predictions_path, predictions, fmt="%.10f")
    np.savetxt(weights_path, W, fmt="%.10f")

if __name__ == "__main__":
    main()