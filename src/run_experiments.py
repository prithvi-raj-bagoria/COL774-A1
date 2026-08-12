import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"

PYTHON = sys.executable

TRAIN = DATA / "e4_hr_train_downsampled.csv"
TEST = DATA / "e4_hr_test_downsampled.csv"
FOLDS = DATA / "train_5fold.txt"
REGULARIZATION = DATA / "regularization.txt"

PRED_A = ROOT / "predictions_a.txt"
WEIGHTS_A = ROOT / "weights_a.txt"

PRED_B = ROOT / "predictions_b.txt"
WEIGHTS_B = ROOT / "weights_b.txt"
BEST_LAMBDA = ROOT / "bestlambda.txt"
CV_ERRORS = ROOT / "crossvalidation_errors.txt"

PRED_C = ROOT / "predictions_c.txt"


def run(command, description):
    print()
    print("=" * 80)
    print(description)
    print("=" * 80)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def load_test_target():
    df = pd.read_csv(TEST)

    if "hr" not in df.columns:
        raise ValueError(
            "The supplied test CSV does not contain 'hr'."
        )

    return df["hr"].to_numpy(dtype=np.float64)


def metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Expected {len(y_true)} predictions, "
            f"got {len(y_pred)}"
        )

    mean_y = np.mean(y_true)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    mse = np.mean(
        (y_true - y_pred) ** 2
    )

    nmae = (
        np.sum(np.abs(y_true - y_pred))
        /
        np.sum(np.abs(y_true - mean_y))
    )

    nmse = (
        np.sum((y_true - y_pred) ** 2)
        /
        np.sum((y_true - mean_y) ** 2)
    )

    return mae, mse, nmae, nmse


def evaluate(prediction_path):
    prediction_path = Path(prediction_path)

    y_true = load_test_target()

    y_pred = np.loadtxt(
        prediction_path,
        dtype=np.float64
    ).reshape(-1)

    return metrics(y_true, y_pred)


def run_a():
    run(
        [
            PYTHON,
            str(SRC / "part_a.py"),
            str(TRAIN),
            str(TEST),
            str(PRED_A),
            str(WEIGHTS_A),
        ],
        "Running Part (a)",
    )


def run_b():
    run(
        [
            PYTHON,
            str(SRC / "part_b.py"),
            str(TRAIN),
            str(TEST),
            str(FOLDS),
            str(REGULARIZATION),
            str(PRED_B),
            str(WEIGHTS_B),
            str(BEST_LAMBDA),
            str(CV_ERRORS),
        ],
        "Running Part (b)",
    )


def run_c():
    run(
        [
            PYTHON,
            str(SRC / "part_c.py"),
            str(TRAIN),
            str(TEST),
            str(PRED_C),
        ],
        "Running Part (c)",
    )


def print_result(name, prediction_path):
    mae, mse, nmae, nmse = evaluate(
        prediction_path
    )

    print()
    print(f"{name}")
    print("-" * 60)
    print(f"MAE  : {mae:.10f}")
    print(f"MSE  : {mse:.10f}")
    print(f"NMAE : {nmae:.10f}")
    print(f"NMSE : {nmse:.10f}")

    return {
        "name": name,
        "mae": mae,
        "mse": mse,
        "nmae": nmae,
        "nmse": nmse,
    }


def run_one(mode):
    if mode == "a":
        run_a()
        return print_result(
            "Part (a) - OLS",
            PRED_A,
        )

    if mode == "b":
        run_b()
        return print_result(
            "Part (b) - Ridge",
            PRED_B,
        )

    if mode == "c":
        run_c()
        return print_result(
            "Part (c) - Feature Engineering",
            PRED_C,
        )

    raise ValueError(
        f"Unknown mode: {mode}"
    )


def print_comparison(results):
    print()
    print("=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)

    print(
        f"{'Model':<32}"
        f"{'MAE':>14}"
        f"{'MSE':>14}"
        f"{'NMAE':>14}"
        f"{'NMSE':>14}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['name']:<32}"
            f"{result['mae']:>14.6f}"
            f"{result['mse']:>14.6f}"
            f"{result['nmae']:>14.6f}"
            f"{result['nmse']:>14.6f}"
        )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python src/run_experiments.py a\n"
            "  python src/run_experiments.py b\n"
            "  python src/run_experiments.py c\n"
            "  python src/run_experiments.py all"
        )

    mode = sys.argv[1].lower()

    if mode in {"a", "b", "c"}:
        run_one(mode)
        return

    if mode == "all":
        results = []

        # Sequential execution is intentional.
        # Part (b) is memory-heavy.
        results.append(run_one("a"))
        results.append(run_one("b"))
        results.append(run_one("c"))

        print_comparison(results)
        return

    raise SystemExit(
        f"Unknown mode '{mode}'. "
        "Use a, b, c, or all."
    )


if __name__ == "__main__":
    main()
