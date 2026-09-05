"""
COL774 Assignment 1 - Part 2(b): Class Imbalance
================================================
Usage:
    python3 part_b.py <train_csv> <test_csv> <method> <pred_file> <weights_file>

method: baseline | classweight | classweight2 | focal
"""

import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & Hyperparameters
# ---------------------------------------------------------------------------
T_ada, lr_ada, B_ada, eps = 200, 0.3, 32, 1e-8
SEED = 774
NUM_CLASSES = 3

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    drop_train = {"release_id", "label"}
    feature_cols = [c for c in train_df.columns if c not in drop_train]

    X_train_raw = train_df[feature_cols].values.astype(np.float64)
    Y_train     = train_df["label"].values.astype(np.int64)

    drop_test = {"release_id"}
    X_test_raw = test_df[feature_cols].values.astype(np.float64)

    mu  = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train_raw - mu) / std
    X_test  = (X_test_raw  - mu) / std

    return X_train, Y_train, X_test

def one_hot(Y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    n = len(Y)
    Y_oh = np.zeros((n, num_classes), dtype=np.float64)
    Y_oh[np.arange(n), Y] = 1.0
    return Y_oh

# ---------------------------------------------------------------------------
# Softmax & Output Helpers
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    shifted = np.clip(shifted, -60.0, 0.0)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def write_weights(W: np.ndarray, b: np.ndarray, path: str):
    with open(path, "w") as f:
        f.write(",".join(f"{v:.17g}" for v in b) + "\n")
        for row in W:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")

def write_predictions(X_test: np.ndarray, W: np.ndarray, b: np.ndarray, path: str):
    logits = X_test @ W + b
    P = softmax(logits)
    with open(path, "w") as f:
        for row in P:
            f.write(",".join(f"{v:.17g}" for v in row) + "\n")

# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_part_b(method: str, X: np.ndarray, Y: np.ndarray):
    n, d = X.shape
    Y_oh = one_hot(Y)
    W = np.zeros((d, NUM_CLASSES), dtype=np.float64)
    b = np.zeros(NUM_CLASSES,      dtype=np.float64)
    GW = np.zeros_like(W)
    Gb = np.zeros_like(b)

    rng = np.random.default_rng(SEED)

    # Class counts
    counts = np.bincount(Y, minlength=NUM_CLASSES)
    alpha = n / (3.0 * counts)
    alpha_yi = alpha[Y]  # shape (n,)

    if method == "baseline":
        batch_weights = np.ones(n, dtype=np.float64)
    elif method == "classweight":
        batch_weights = alpha_yi
    elif method == "classweight2":
        batch_weights = alpha_yi ** 0.3
    elif method == "focal":
        alpha_prime = alpha_yi ** 0.5
        gamma = 2.0
    else:
        raise ValueError(f"Unknown method {method}")

    losses = []

    for epoch in range(T_ada):
        order = rng.permutation(n)
        for start in range(0, n, B_ada):
            idx = order[start: start + B_ada]
            X_b = X[idx]
            Y_b = Y_oh[idx]
            logits = X_b @ W + b
            P_b = softmax(logits)
            
            diff = P_b - Y_b

            if method in ["baseline", "classweight", "classweight2"]:
                w_b = batch_weights[idx]
                # Normalize within batch
                w_b_norm = w_b / w_b.sum()
                w_b_norm = w_b_norm.reshape(-1, 1)  # (|B|, 1)
                
                gW = X_b.T @ (diff * w_b_norm)
                gb = (diff * w_b_norm).sum(axis=0)
                
            elif method == "focal":
                p_t = P_b[np.arange(len(idx)), Y[idx]]
                p_t = np.clip(p_t, 1e-12, 1.0 - 1e-12)
                
                a_prime = alpha_prime[idx]
                
                # c_i = alpha'_t * (1 - p_t)^(gamma-1) * [(1 - p_t) - gamma * p_t * log(p_t)]
                c_i = a_prime * (1 - p_t)**(gamma - 1) * ((1 - p_t) - gamma * p_t * np.log(p_t))
                # Average over mini-batch
                c_i_norm = (c_i / len(idx)).reshape(-1, 1)
                
                gW = X_b.T @ (diff * c_i_norm)
                gb = (diff * c_i_norm).sum(axis=0)

            GW += gW * gW
            Gb += gb * gb

            W -= lr_ada * gW / (np.sqrt(GW) + eps)
            b -= lr_ada * gb / (np.sqrt(Gb) + eps)

        # Full loss calculation after epoch
        logits_full = X @ W + b
        P_full = softmax(logits_full)
        if method in ["baseline", "classweight", "classweight2"]:
            p_t_full = P_full[np.arange(n), Y]
            p_t_full = np.clip(p_t_full, 1e-15, 1.0)
            ce_losses = -np.log(p_t_full)
            epoch_loss = np.sum(batch_weights * ce_losses) / np.sum(batch_weights)
        elif method == "focal":
            p_t_full = P_full[np.arange(n), Y]
            p_t_full = np.clip(p_t_full, 1e-12, 1.0 - 1e-12)
            focal_losses = alpha_prime * (1 - p_t_full)**gamma * (-np.log(p_t_full))
            epoch_loss = np.mean(focal_losses)

        losses.append(epoch_loss)

    return W, b, losses

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 6:
        print("Usage: python3 part_b.py <train_csv> <test_csv> <method> <pred_file> <weights_file>")
        sys.exit(1)

    train_csv, test_csv, method, pred_file, weights_file = sys.argv[1:]

    X_train, Y_train, X_test = load_data(train_csv, test_csv)
    print(f"[{method}] train={len(X_train)} examples, test={len(X_test)}")

    W, b, losses = train_part_b(method, X_train, Y_train)

    print(f"[{method}] Final train loss: {losses[-1]:.6f}")

    write_weights(W, b, weights_file)
    write_predictions(X_test, W, b, pred_file)
    print(f"[{method}] Wrote outputs")

if __name__ == "__main__":
    main()

