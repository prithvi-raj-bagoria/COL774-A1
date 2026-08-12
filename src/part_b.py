import sys
import numpy as np
import pandas as pd


def ridge_weights(gram, rhs, lam):
    """
    Compute ridge regression parameters using
        W = (X^T X + lambda R)^(-1) X^T Y
    where
        R = diag(0, 1, 1, ..., 1)
    The first parameter is the intercept and is not regularized.
    """
    A = gram.copy()

    # Add lambda only to the feature coefficients.
    # A[0,0] corresponds to the intercept, so it is untouched.
    A[1:, 1:] += lam * np.eye(A.shape[0] - 1)

    # Assignment explicitly requires numpy.linalg.inv.
    W = np.linalg.inv(A) @ rhs

    return W


def nmse(y_true, y_pred):
    """
    Fold-wise NMSE from the assignment:
        sum((y - y_hat)^2)
        -------------------
        sum((y - y_fold_mean)^2)
    """
    y_mean = np.mean(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_mean) ** 2)
    return numerator / denominator


def parse_fold_file(path, n):
    """
    Parse the supplied train_5fold.txt. Supports both:
      1. Your local format: fold_ends = [32161, 64993, 97079, 127934, 158986]
      2. PDF format fallback: Line-separated fold numbers for each training example
    """
    with open(path, "r") as f:
        content = f.read().strip()

    left = content.find("[")
    right = content.find("]")

    # Case 1: 'fold_ends = [...]' format
    if left != -1 and right != -1:
        values = content[left + 1:right].split(",")
        fold_ends = [int(v.strip()) for v in values if v.strip()]

        if len(fold_ends) != 5:
            raise ValueError(f"Expected 5 fold endpoints, got {len(fold_ends)}")
        if fold_ends[-1] != n:
            raise ValueError(f"Final fold endpoint {fold_ends[-1]} != {n}")

        folds = []
        start = 0
        for end in fold_ends:
            if end <= start or end > n:
                raise ValueError(f"Invalid fold boundary: {start} -> {end}")
            # Slice rather than integer indexing gives a memory-efficient VIEW
            folds.append(slice(start, end))
            start = end
        return folds

    # Case 2: Fallback to individual line-by-line fold IDs (strict PDF format)
    tokens = content.replace(",", " ").split()
    if len(tokens) == n:
        fold_ids = np.array([int(t) for t in tokens], dtype=int)
        return [np.where(fold_ids == k)[0] for k in range(5)]

    raise ValueError(f"Unable to parse fold structures from {path} with n={n}")


def make_augmented_statistics(X, y):
    """
    Compute X_aug^T X_aug and X_aug^T y directly.
    """
    n, m = X.shape

    gram = np.empty((m + 1, m + 1), dtype=np.float64)

    gram[0, 0] = n
    feature_sums = np.sum(X, axis=0)
    gram[0, 1:] = feature_sums
    gram[1:, 0] = feature_sums
    gram[1:, 1:] = X.T @ X

    rhs = np.empty(m + 1, dtype=np.float64)
    rhs[0] = np.sum(y)
    rhs[1:] = X.T @ y

    return gram, rhs


def main():
    if len(sys.argv) != 9:
        raise SystemExit(
            "Usage:\n"
            "python3 part_b.py train.csv test.csv folds.txt "
            "regularization.txt predictions.txt weights.txt "
            "bestlambda.txt crossvalidation_errors.txt"
        )

    train_path = sys.argv[1]
    test_path = sys.argv[2]
    folds_path = sys.argv[3]
    regularization_path = sys.argv[4]
    predictions_path = sys.argv[5]
    weights_path = sys.argv[6]
    bestlambda_path = sys.argv[7]
    cv_errors_path = sys.argv[8]

    # Load TRAIN only.
    train_df = pd.read_csv(train_path)

    if "hr" not in train_df.columns:
        raise ValueError("Target column 'hr' not found in training dataset.")

    # Get feature columns preserving exact order, removing 'hr'
    feature_columns = [col for col in train_df.columns if col != "hr"]

    if len(feature_columns) != 1640:
        raise ValueError(f"Expected 1640 features, found {len(feature_columns)}")

    X = train_df[feature_columns].to_numpy(dtype=np.float64)
    y = train_df["hr"].to_numpy(dtype=np.float64)
    del train_df

    n, m = X.shape
    print(f"Training examples : {n}")
    print(f"Features          : {m}")

    # Parse folds
    folds = parse_fold_file(folds_path, n)

    # Read candidate lambdas (Store both raw string and float value)
    lambda_entries = []
    with open(regularization_path, "r") as f:
        for line in f:
            raw = line.strip()
            if raw:
                lambda_entries.append((raw, float(raw)))

    if not lambda_entries:
        raise ValueError("regularization.txt contains no lambda values.")

    # Compute FULL augmented sufficient statistics once
    print("\nComputing full training statistics...")
    full_gram, full_rhs = make_augmented_statistics(X, y)

    # Compute statistics for each validation fold
    fold_grams = []
    fold_rhss = []

    print("Computing fold statistics...")
    for k, fold in enumerate(folds):
        X_val = X[fold]
        y_val = y[fold]
        gram_k, rhs_k = make_augmented_statistics(X_val, y_val)
        fold_grams.append(gram_k)
        fold_rhss.append(rhs_k)
        print(f"  Fold {k} done")

    # Five-fold cross-validation
    cv_results = []
    best_lambda_raw = None
    best_lambda_val = None
    best_cv_nmse = np.inf

    print("\nStarting cross-validation...\n")

    for raw_lam, lam_val in lambda_entries:
        print(f"lambda = {raw_lam}")
        fold_errors = []

        for k, fold in enumerate(folds):
            fold_gram = fold_grams[k]
            fold_rhs = fold_rhss[k]

            # all data - validation fold
            train_gram = full_gram - fold_gram
            train_rhs = full_rhs - fold_rhs

            W = ridge_weights(train_gram, train_rhs, lam_val)

            X_val = X[fold]
            y_val = y[fold]

            y_pred = W[0] + X_val @ W[1:]
            error = nmse(y_val, y_pred)
            fold_errors.append(error)

            print(f"    Fold {k}: NMSE = {error:.10f}")

            del train_gram
            del train_rhs
            del W

        cv_nmse = np.mean(fold_errors)
        cv_results.append((raw_lam, cv_nmse))

        print(f"    CVNMSE = {cv_nmse:.10f}\n")

        # Strict < preserves the first lambda in case of exact tie
        if cv_nmse < best_cv_nmse:
            best_cv_nmse = cv_nmse
            best_lambda_raw = raw_lam
            best_lambda_val = lam_val

    # Final model on all training examples
    print("Selected lambda:", best_lambda_raw)
    print("Best CVNMSE    :", best_cv_nmse)

    final_W = ridge_weights(full_gram, full_rhs, best_lambda_val)

    # Free memory
    del X
    del y
    del full_gram
    del full_rhs
    del fold_grams
    del fold_rhss
    del folds

    # Load TEST
    print("\nLoading test data...")
    test_df = pd.read_csv(test_path)
    X_test = test_df[feature_columns].to_numpy(dtype=np.float64)
    del test_df

    if X_test.shape[1] != 1640:
        raise ValueError(f"Expected 1640 test features, found {X_test.shape[1]}")

    predictions = final_W[0] + X_test @ final_W[1:]

    # Save output files
    np.savetxt(predictions_path, predictions, fmt="%.10f")
    np.savetxt(weights_path, final_W, fmt="%.10f")

    # Save best lambda using raw string
    with open(bestlambda_path, "w") as f:
        f.write(f"{best_lambda_raw}\n")

    # Save CV errors using raw string for lambda and matching PDF decimals
    with open(cv_errors_path, "w") as f:
        for raw_lam, error in cv_results:
            f.write(f"{raw_lam},{error:.6f}\n")

    print("\nDone.")


if __name__ == "__main__":
    main()