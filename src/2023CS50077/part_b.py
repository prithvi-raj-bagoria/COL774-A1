import sys
import numpy as np
import pandas as pd

def ridge_weights(X_aug, y, lam):
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ y
    R = np.eye(XtX.shape[0])
    R[0, 0] = 0.0
    return np.linalg.inv(XtX + lam * R) @ XtY

def nmse(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_mean) ** 2)

def parse_fold_file(path, n):
    with open(path, "r") as f:
        content = f.read().strip()
    left, right = content.find("["), content.find("]")
    if left != -1 and right != -1:
        fold_ends = [int(v.strip()) for v in content[left + 1:right].split(",") if v.strip()]
        folds, start = [], 0
        for end in fold_ends:
            folds.append(slice(start, end))
            start = end
        return folds
    tokens = content.replace(",", " ").split()
    if len(tokens) == n:
        fold_ids = np.array([int(t) for t in tokens], dtype=int)
        return [np.where(fold_ids == k)[0] for k in range(5)]
    raise ValueError("Unable to parse folds.")

def main():
    if len(sys.argv) != 9:
        raise SystemExit(
            "Usage: python3 part_b.py train.csv test.csv folds.txt "
            "regularization.txt predictions.txt weights.txt "
            "bestlambda.txt crossvalidation_errors.txt"
        )
    train_path, test_path, folds_path, regularization_path = sys.argv[1:5]
    predictions_path, weights_path, bestlambda_path, cv_errors_path = sys.argv[5:9]

    train_df = pd.read_csv(train_path)
    feature_columns = [col for col in train_df.columns if col != "hr"]
    X = train_df[feature_columns].to_numpy(dtype=np.float64)
    y = train_df["hr"].to_numpy(dtype=np.float64)
    del train_df

    n = X.shape[0]
    folds = parse_fold_file(folds_path, n)
    X_aug = np.hstack((np.ones((n, 1)), X))
    del X

    lambda_entries = []
    with open(regularization_path, "r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                lambda_entries.append((raw, float(raw)))

    cv_results = []
    best_lambda_raw, best_lambda_val, best_cv_nmse = None, None, np.inf

    for raw_lam, lam_val in lambda_entries:
        fold_errors = []
        for k, fold in enumerate(folds):
            mask = np.ones(n, dtype=bool)
            mask[fold] = False
            X_train_aug, y_train = X_aug[mask], y[mask]
            X_val_aug, y_val = X_aug[fold], y[fold]
            W = ridge_weights(X_train_aug, y_train, lam_val)
            y_pred = X_val_aug @ W
            fold_errors.append(nmse(y_val, y_pred))
            del X_train_aug, y_train, X_val_aug, y_val, W, y_pred
        cv_nmse = np.mean(fold_errors)
        cv_results.append((raw_lam, cv_nmse))
        if cv_nmse < best_cv_nmse:
            best_cv_nmse = cv_nmse
            best_lambda_raw = raw_lam
            best_lambda_val = lam_val

    final_W = ridge_weights(X_aug, y, best_lambda_val)
    del X_aug, y

    test_df = pd.read_csv(test_path)
    X_test = test_df[feature_columns].to_numpy(dtype=np.float64)
    del test_df
    X_test_aug = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    predictions = X_test_aug @ final_W
    del X_test, X_test_aug

    np.savetxt(predictions_path, predictions, fmt="%.10f")
    np.savetxt(weights_path, final_W, fmt="%.10f")
    with open(bestlambda_path, "w") as f:
        f.write(f"{best_lambda_raw}\n")
    with open(cv_errors_path, "w") as f:
        for raw_lam, error in cv_results:
            f.write(f"{raw_lam},{error:.6f}\n")

if __name__ == "__main__":
    main()