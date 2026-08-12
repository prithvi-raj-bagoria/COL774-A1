import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SRC_DIR = ROOT / "src"

TRAIN = DATA_DIR / "e4_hr_train_downsampled.csv"
TEST = DATA_DIR / "e4_hr_test_downsampled.csv"
FOLDS = DATA_DIR / "train_5fold.txt"
REGULARIZATION = DATA_DIR / "regularization.txt"

PYTHON = sys.executable


# ============================================================
# Output files
# ============================================================

PRED_A = ROOT / "predictions_a.txt"

PRED_B = ROOT / "predictions_b.txt"
WEIGHTS_B = ROOT / "weights_b.txt"
BEST_LAMBDA = ROOT / "bestlambda.txt"
CV_ERRORS = ROOT / "crossvalidation_errors.txt"

PRED_C = ROOT / "predictions_c.txt"
PRED_C_BVP = ROOT / "predictions_c_bvp.txt"


# ============================================================
# Run command
# ============================================================

def run_command(command, description):
    print()
    print("=" * 80)
    print(description)
    print("=" * 80)
    print("Command:")
    print(" ".join(str(x) for x in command))
    print()

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


# ============================================================
# Load true targets
# ============================================================

def load_test_target():
    test = pd.read_csv(TEST)

    if "hr" not in test.columns:
        raise ValueError(
            "The supplied test CSV does not contain 'hr'."
        )

    return test["hr"].to_numpy(dtype=np.float64)


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Prediction count mismatch: "
            f"expected {len(y_true)}, got {len(y_pred)}"
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


# ============================================================
# Evaluate one prediction file
# ============================================================

def evaluate_prediction(
    prediction_path,
    model_name,
    y_true,
):
    prediction_path = Path(prediction_path)

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {prediction_path}"
        )

    y_pred = np.loadtxt(
        prediction_path,
        dtype=np.float64
    )

    y_pred = np.asarray(y_pred).reshape(-1)

    mae, mse, nmae, nmse = calculate_metrics(
        y_true,
        y_pred
    )

    return {
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "nmae": nmae,
        "nmse": nmse,
    }


# ============================================================
# Part A
# ============================================================

def run_a(y_true):
    run_command(
        [
            PYTHON,
            str(SRC_DIR / "part_a.py"),
            str(TRAIN),
            str(TEST),
            str(PRED_A),
            str(ROOT / "weights_a.txt"),
        ],
        "Running Part (a) from scratch",
    )

    return evaluate_prediction(
        PRED_A,
        "Part A - OLS",
        y_true,
    )


# ============================================================
# Part B
# ============================================================

def run_b(y_true):
    run_command(
        [
            PYTHON,
            str(SRC_DIR / "part_b.py"),
            str(TRAIN),
            str(TEST),
            str(FOLDS),
            str(REGULARIZATION),
            str(PRED_B),
            str(WEIGHTS_B),
            str(BEST_LAMBDA),
            str(CV_ERRORS),
        ],
        "Running Part (b) from scratch",
    )

    return evaluate_prediction(
        PRED_B,
        "Part B - Ridge",
        y_true,
    )


# ============================================================
# Part C
# ============================================================

def run_c(y_true):
    run_command(
        [
            PYTHON,
            str(SRC_DIR / "part_c.py"),
            str(TRAIN),
            str(TEST),
            str(PRED_C),
        ],
        "Running Part (c) from scratch",
    )

    return evaluate_prediction(
        PRED_C,
        "Part C - Baseline",
        y_true,
    )


# ============================================================
# Part C BVP experiment
# ============================================================

def run_c_bvp(y_true):
    script = SRC_DIR / "part_c_bvp.py"

    if not script.exists():
        raise FileNotFoundError(
            f"{script} does not exist."
        )

    run_command(
        [
            PYTHON,
            str(script),
            str(TRAIN),
            str(TEST),
            str(PRED_C_BVP),
        ],
        "Running Part (c) BVP experiment from scratch",
    )

    return evaluate_prediction(
        PRED_C_BVP,
        "Part C - BVP Enhanced",
        y_true,
    )


# ============================================================
# Print result
# ============================================================

def print_result(result):
    print()
    print("=" * 80)
    print(result["model"])
    print("=" * 80)

    print(f"MAE  : {result['mae']:.10f}")
    print(f"MSE  : {result['mse']:.10f}")
    print(f"NMAE : {result['nmae']:.10f}")
    print(f"NMSE : {result['nmse']:.10f}")


# ============================================================
# Comparison table
# ============================================================

def print_table(results):
    print()
    print("=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    print(
        f"{'Model':<30}"
        f"{'MAE':>14}"
        f"{'MSE':>16}"
        f"{'NMAE':>14}"
        f"{'NMSE':>14}"
    )

    print("-" * 100)

    for r in results:
        print(
            f"{r['model']:<30}"
            f"{r['mae']:>14.6f}"
            f"{r['mse']:>16.6f}"
            f"{r['nmae']:>14.6f}"
            f"{r['nmse']:>14.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            """
Usage:

    python src/run_experiments.py a
    python src/run_experiments.py b
    python src/run_experiments.py c
    python src/run_experiments.py c_bvp
    python src/run_experiments.py all

The selected experiment is always rerun from scratch.
"""
        )

    mode = sys.argv[1].lower()

    valid_modes = {
        "a",
        "b",
        "c",
        "c_bvp",
        "all",
    }

    if mode not in valid_modes:
        raise SystemExit(
            f"Unknown mode '{mode}'. "
            f"Choose from {sorted(valid_modes)}."
        )

    y_true = load_test_target()

    results = []

    if mode == "a":
        result = run_a(y_true)
        print_result(result)

    elif mode == "b":
        result = run_b(y_true)
        print_result(result)

    elif mode == "c":
        result = run_c(y_true)
        print_result(result)

    elif mode == "c_bvp":
        result = run_c_bvp(y_true)
        print_result(result)

    elif mode == "all":

        # Run sequentially so we don't launch multiple
        # memory-heavy models simultaneously.
        results.append(
            run_a(y_true)
        )

        results.append(
            run_b(y_true)
        )

        results.append(
            run_c(y_true)
        )

        results.append(
            run_c_bvp(y_true)
        )

        print_table(results)


if __name__ == "__main__":
    main()