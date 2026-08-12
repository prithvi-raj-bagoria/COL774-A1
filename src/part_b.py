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

    `gram` and `rhs` are already the augmented quantities:

        gram = X_aug^T X_aug
        rhs  = X_aug^T y
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
    Parse the actual supplied train_5fold.txt:

        fold_ends = [32161, 64993, 97079, 127934, 158986]

    The folds are:

        F0 = [0, 32161)
        F1 = [32161, 64993)
        F2 = [64993, 97079)
        F3 = [97079, 127934)
        F4 = [127934, 158986)
    """

    with open(path, "r") as f:
        content = f.read().strip()

    left = content.find("[")
    right = content.find("]")

    if left == -1 or right == -1:
        raise ValueError("Could not parse fold_ends from folds file.")

    values = content[left + 1:right].split(",")

    fold_ends = [
        int(v.strip())
        for v in values
        if v.strip()
    ]

    if len(fold_ends) != 5:
        raise ValueError(
            f"Expected 5 fold endpoints, got {len(fold_ends)}"
        )

    if fold_ends[-1] != n:
        raise ValueError(
            f"Final fold endpoint {fold_ends[-1]} "
            f"does not equal number of training examples {n}"
        )

    folds = []

    start = 0

    for end in fold_ends:

        if end <= start or end > n:
            raise ValueError(
                f"Invalid fold boundary: {start} -> {end}"
            )

        # Slice rather than integer indexing:
        # this gives a VIEW instead of making a copy.
        folds.append(slice(start, end))

        start = end

    return folds


def make_augmented_statistics(X, y):
    """
    Instead of explicitly creating

        X_aug = [1 | X]

    compute X_aug^T X_aug and X_aug^T y directly.

    This avoids creating another ~2 GB matrix.

    For

        X_aug = [1 | X]

    we have

        X_aug^T X_aug =
        [ n       sum(X)   ]
        [ sum(X)^T X^T X  ]

    and

        X_aug^T y =
        [ sum(y) ]
        [ X^T y  ].
    """

    n, m = X.shape

    gram = np.empty(
        (m + 1, m + 1),
        dtype=np.float64
    )

    # Upper-left: intercept-intercept
    gram[0, 0] = n

    # Intercept-feature terms
    feature_sums = np.sum(X, axis=0)

    gram[0, 1:] = feature_sums
    gram[1:, 0] = feature_sums

    # Feature-feature terms
    gram[1:, 1:] = X.T @ X

    # X_aug^T y
    rhs = np.empty(m + 1, dtype=np.float64)

    rhs[0] = np.sum(y)
    rhs[1:] = X.T @ y

    return gram, rhs


def main():

    # ---------------------------------------------------------
    # Required interface
    #
    # python3 part_b.py train.csv test.csv folds.txt
    #            regularization.txt predictions.txt weights.txt
    #            bestlambda.txt crossvalidation_errors.txt
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Load TRAIN only.
    #
    # We deliberately do NOT load test yet because CV does
    # not require test data.
    # ---------------------------------------------------------

    train_df = pd.read_csv(train_path)

    if train_df.columns[-1] != "hr":
        raise ValueError(
            f"Expected final training column 'hr', "
            f"found '{train_df.columns[-1]}'"
        )

    feature_columns = train_df.columns[:-1]

    if len(feature_columns) != 1640:
        raise ValueError(
            f"Expected 1640 features, found {len(feature_columns)}"
        )

    # Convert to NumPy.
    X = train_df[feature_columns].to_numpy(
        dtype=np.float64
    )

    y = train_df["hr"].to_numpy(
        dtype=np.float64
    )

    # The DataFrame is no longer required.
    del train_df

    n, m = X.shape

    print(f"Training examples : {n}")
    print(f"Features           : {m}")

    # ---------------------------------------------------------
    # Parse folds
    # ---------------------------------------------------------

    folds = parse_fold_file(
        folds_path,
        n
    )

    print("\nFold sizes:")

    for k, fold in enumerate(folds):
        size = fold.stop - fold.start
        print(f"  Fold {k}: {size}")

    # ---------------------------------------------------------
    # Read candidate lambdas
    # ---------------------------------------------------------

    with open(regularization_path, "r") as f:
        lambdas = [
            float(line.strip())
            for line in f
            if line.strip()
        ]

    if not lambdas:
        raise ValueError(
            "regularization.txt contains no lambda values."
        )

    print("\nLambda candidates:")
    print(lambdas)

    # ---------------------------------------------------------
    # Compute FULL augmented sufficient statistics once.
    #
    # We do NOT construct X_aug.
    #
    # This is the key memory optimization.
    # ---------------------------------------------------------

    print("\nComputing full training statistics...")

    full_gram, full_rhs = make_augmented_statistics(
        X,
        y
    )

    # ---------------------------------------------------------
    # Compute statistics for each validation fold.
    #
    # Since folds are contiguous, X[fold] is a VIEW.
    # No giant training-fold copy is created.
    #
    # Only a ~1641 x 1641 matrix is stored for each fold.
    # ---------------------------------------------------------

    fold_grams = []
    fold_rhss = []

    print("Computing fold statistics...")

    for k, fold in enumerate(folds):

        X_val = X[fold]
        y_val = y[fold]

        gram_k, rhs_k = make_augmented_statistics(
            X_val,
            y_val
        )

        fold_grams.append(gram_k)
        fold_rhss.append(rhs_k)

        print(f"  Fold {k} done")

    # ---------------------------------------------------------
    # Five-fold cross-validation
    # ---------------------------------------------------------

    cv_results = []

    best_lambda = None
    best_cv_nmse = np.inf

    print("\nStarting cross-validation...\n")

    for lam in lambdas:

        print(f"lambda = {lam}")

        fold_errors = []

        for k, fold in enumerate(folds):

            # Validation statistics
            fold_gram = fold_grams[k]
            fold_rhs = fold_rhss[k]

            # Training statistics:
            #
            # all data - validation fold
            train_gram = full_gram - fold_gram
            train_rhs = full_rhs - fold_rhs

            # Train ridge
            W = ridge_weights(
                train_gram,
                train_rhs,
                lam
            )

            # Validation data is a VIEW.
            X_val = X[fold]
            y_val = y[fold]

            # Do NOT create X_val_aug.
            #
            # prediction = intercept + X_val @ coefficients
            y_pred = (
                W[0]
                + X_val @ W[1:]
            )

            error = nmse(
                y_val,
                y_pred
            )

            fold_errors.append(error)

            print(
                f"    Fold {k}: "
                f"NMSE = {error:.10f}"
            )

            del train_gram
            del train_rhs
            del W

        cv_nmse = np.mean(fold_errors)

        cv_results.append(
            (lam, cv_nmse)
        )

        print(
            f"    CVNMSE = {cv_nmse:.10f}"
        )

        # Strict < preserves the first lambda in case of
        # an exact tie, as required by the assignment.
        if cv_nmse < best_cv_nmse:
            best_cv_nmse = cv_nmse
            best_lambda = lam

        print()

    # ---------------------------------------------------------
    # Final model:
    #
    # Retrain on ALL training examples.
    # ---------------------------------------------------------

    print("Selected lambda:", best_lambda)
    print("Best CVNMSE    :", best_cv_nmse)

    final_W = ridge_weights(
        full_gram,
        full_rhs,
        best_lambda
    )

    # ---------------------------------------------------------
    # We no longer need the training feature matrix.
    #
    # Free ~2 GB before loading test data.
    # ---------------------------------------------------------

    del X
    del y
    del full_gram
    del full_rhs
    del fold_grams
    del fold_rhss
    del folds

    # ---------------------------------------------------------
    # Load TEST only AFTER training is finished.
    # ---------------------------------------------------------

    print("\nLoading test data...")

    test_df = pd.read_csv(test_path)

    X_test = test_df[feature_columns].to_numpy(
        dtype=np.float64
    )

    del test_df

    if X_test.shape[1] != 1640:
        raise ValueError(
            f"Expected 1640 test features, "
            f"found {X_test.shape[1]}"
        )

    # ---------------------------------------------------------
    # Test prediction without constructing X_test_aug.
    # ---------------------------------------------------------

    predictions = (
        final_W[0]
        + X_test @ final_W[1:]
    )

    # ---------------------------------------------------------
    # predictions.txt
    # ---------------------------------------------------------

    np.savetxt(
        predictions_path,
        predictions,
        fmt="%.10f"
    )

    # ---------------------------------------------------------
    # weights.txt
    #
    # Intercept first.
    # ---------------------------------------------------------

    np.savetxt(
        weights_path,
        final_W,
        fmt="%.10f"
    )

    # ---------------------------------------------------------
    # bestlambda.txt
    # ---------------------------------------------------------

    with open(bestlambda_path, "w") as f:
        f.write(f"{best_lambda}\n")

    # ---------------------------------------------------------
    # crossvalidation_errors.txt
    #
    # Preserve lambda order.
    # ---------------------------------------------------------

    with open(cv_errors_path, "w") as f:

        for lam, error in cv_results:
            f.write(
                f"{lam},{error:.10f}\n"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()