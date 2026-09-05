"""
COL774 Assignment 1 – Part 2(a): Gradient Descent Variants
===========================================================
Usage:
    python3 part_a.py <train_csv> <test_csv> <method> <pred_file> <weights_file>

method: full_batch | mini_batch | sgd | adagrad
"""

import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HYPERPARAMS = {
    "full_batch": {"epochs": 500, "lr": 0.3,    "batch": None},
    "mini_batch": {"epochs": 200, "lr": 0.03,   "batch": 32},
    "sgd":        {"epochs": 30,  "lr": 0.001,  "batch": 1},
    "adagrad":    {"epochs": 200, "lr": 0.3,    "batch": 32, "eps": 1e-8},
}
SEED = 774
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Softmax (numerically stable, clipped)
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    """
    logits: (n, 3)  float64
    Returns probabilities (n, 3).
    Stability: subtract row-wise max, clip shifted logits to [-60, 0].
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    shifted = np.clip(shifted, -60.0, 0.0)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Loss & gradients
# ---------------------------------------------------------------------------
def cross_entropy_loss(P: np.ndarray, Y: np.ndarray) -> float:
    """Mean cross-entropy loss. P, Y: (n, 3)."""
    eps = 1e-15
    return -np.mean(np.sum(Y * np.log(np.clip(P, eps, 1.0)), axis=1))


def compute_gradients(X_b: np.ndarray, P_b: np.ndarray, Y_b: np.ndarray):
    """
    Gradients of mean cross-entropy w.r.t. W and b for a batch.
    X_b: (|B|, 78), P_b: (|B|, 3), Y_b: (|B|, 3)
    Returns gW (78,3), gb (3,)
    """
    diff = P_b - Y_b               # (|B|, 3)
    gW = X_b.T @ diff / len(X_b)  # (78, 3)
    gb = diff.mean(axis=0)         # (3,)
    return gW, gb


# ---------------------------------------------------------------------------
# Full-dataset forward pass (for epoch loss logging)
# ---------------------------------------------------------------------------
def full_loss(X: np.ndarray, Y_oh: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
    logits = X @ W + b
    P = softmax(logits)
    return cross_entropy_loss(P, Y_oh)


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_data(train_path: str, test_path: str):
    """
    Returns:
        X_train (n_train, 78) float64, standardised
        Y_train (n_train,)    int64
        X_test  (n_test,  78) float64, standardised with train stats
        feature_cols          list of 78 column names
    """
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Identify feature columns: everything except release_id and label
    drop_train = {"release_id", "label"}
    feature_cols = [c for c in train_df.columns if c not in drop_train]

    X_train_raw = train_df[feature_cols].values.astype(np.float64)
    Y_train     = train_df["label"].values.astype(np.int64)

    drop_test = {"release_id"}
    test_feature_cols = [c for c in test_df.columns if c not in drop_test]
    # Align to same feature columns (in case test has extra cols or different order)
    X_test_raw = test_df[feature_cols].values.astype(np.float64)

    # Standardise using training statistics only
    mu  = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0            # avoid div-by-zero for constant features

    X_train = (X_train_raw - mu) / std
    X_test  = (X_test_raw  - mu) / std

    return X_train, Y_train, X_test, feature_cols


def one_hot(Y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    n = len(Y)
    Y_oh = np.zeros((n, num_classes), dtype=np.float64)
    Y_oh[np.arange(n), Y] = 1.0
    return Y_oh


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_full_batch(X: np.ndarray, Y_oh: np.ndarray, rng: np.random.Generator):
    hp = HYPERPARAMS["full_batch"]
    T, lr = hp["epochs"], hp["lr"]
    n, d  = X.shape

    W = np.zeros((d, NUM_CLASSES), dtype=np.float64)
    b = np.zeros(NUM_CLASSES,      dtype=np.float64)

    losses = []
    for epoch in range(T):
        logits = X @ W + b
        P      = softmax(logits)
        gW, gb = compute_gradients(X, P, Y_oh)
        W     -= lr * gW
        b     -= lr * gb
        # Record loss AFTER the update (matches reference convention)
        losses.append(full_loss(X, Y_oh, W, b))

    return W, b, losses


def _epoch_minibatch_loop(X, Y_oh, W, b, lr, batch_size, rng):
    """One epoch of mini-batch updates; returns updated W, b."""
    n = len(X)
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx   = order[start: start + batch_size]
        X_b   = X[idx]
        Y_b   = Y_oh[idx]
        logits = X_b @ W + b
        P_b    = softmax(logits)
        gW, gb = compute_gradients(X_b, P_b, Y_b)
        W -= lr * gW
        b -= lr * gb
    return W, b


def train_mini_batch(X: np.ndarray, Y_oh: np.ndarray, rng: np.random.Generator):
    hp = HYPERPARAMS["mini_batch"]
    T, lr, B = hp["epochs"], hp["lr"], hp["batch"]
    n, d = X.shape

    W = np.zeros((d, NUM_CLASSES), dtype=np.float64)
    b = np.zeros(NUM_CLASSES,      dtype=np.float64)

    losses = []
    for epoch in range(T):
        W, b = _epoch_minibatch_loop(X, Y_oh, W, b, lr, B, rng)
        losses.append(full_loss(X, Y_oh, W, b))

    return W, b, losses


def train_sgd(X: np.ndarray, Y_oh: np.ndarray, rng: np.random.Generator):
    hp = HYPERPARAMS["sgd"]
    T, lr = hp["epochs"], hp["lr"]
    n, d = X.shape

    W = np.zeros((d, NUM_CLASSES), dtype=np.float64)
    b = np.zeros(NUM_CLASSES,      dtype=np.float64)

    losses = []
    for epoch in range(T):
        W, b = _epoch_minibatch_loop(X, Y_oh, W, b, lr, 1, rng)
        losses.append(full_loss(X, Y_oh, W, b))

    return W, b, losses


def train_adagrad(X: np.ndarray, Y_oh: np.ndarray, rng: np.random.Generator):
    hp = HYPERPARAMS["adagrad"]
    T, lr, B, eps = hp["epochs"], hp["lr"], hp["batch"], hp["eps"]
    n, d = X.shape

    W  = np.zeros((d, NUM_CLASSES), dtype=np.float64)
    b  = np.zeros(NUM_CLASSES,      dtype=np.float64)
    GW = np.zeros_like(W)
    Gb = np.zeros_like(b)

    losses = []
    for epoch in range(T):
        order = rng.permutation(n)
        for start in range(0, n, B):
            idx    = order[start: start + B]
            X_b    = X[idx]
            Y_b    = Y_oh[idx]
            logits = X_b @ W + b
            P_b    = softmax(logits)
            gW, gb = compute_gradients(X_b, P_b, Y_b)

            GW += gW * gW
            Gb += gb * gb

            W -= lr * gW / (np.sqrt(GW) + eps)
            b -= lr * gb / (np.sqrt(Gb) + eps)

        losses.append(full_loss(X, Y_oh, W, b))

    return W, b, losses


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_weights(W: np.ndarray, b: np.ndarray, path: str):
    """
    79 lines: first line = bias (3 comma-separated values),
    next 78 lines = rows of W (3 comma-separated values each).
    """
    with open(path, "w") as f:
        f.write(",".join(f"{v:.17g}" for v in b) + "\n")
        for row in W:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")


def write_predictions(X_test: np.ndarray, W: np.ndarray, b: np.ndarray, path: str):
    """
    One line per test row: p(N), p(A), p(O) comma-separated.
    """
    logits = X_test @ W + b
    P      = softmax(logits)
    with open(path, "w") as f:
        for row in P:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TRAINERS = {
    "full_batch": train_full_batch,
    "mini_batch": train_mini_batch,
    "sgd":        train_sgd,
    "adagrad":    train_adagrad,
}


def main():
    if len(sys.argv) != 6:
        print("Usage: python3 part_a.py <train_csv> <test_csv> <method> <pred_file> <weights_file>")
        sys.exit(1)

    train_csv, test_csv, method, pred_file, weights_file = sys.argv[1:]

    if method not in TRAINERS:
        print(f"Unknown method '{method}'. Choose from: {list(TRAINERS)}")
        sys.exit(1)

    # Load data
    X_train, Y_train, X_test, feature_cols = load_data(train_csv, test_csv)
    Y_oh = one_hot(Y_train)

    n_train, d = X_train.shape
    print(f"[{method}] train={n_train} examples, {d} features, test={len(X_test)}")

    # Single RNG created once before training
    rng = np.random.default_rng(SEED)

    # Train
    trainer = TRAINERS[method]
    W, b, train_losses = trainer(X_train, Y_oh, rng)

    print(f"[{method}] Final train loss: {train_losses[-1]:.6f}")

    # Write outputs
    write_weights(W, b, weights_file)
    write_predictions(X_test, W, b, pred_file)
    print(f"[{method}] Wrote predictions -> {pred_file}")
    print(f"[{method}] Wrote weights     -> {weights_file}")


if __name__ == "__main__":
    main()

