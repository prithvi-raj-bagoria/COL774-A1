# Variables for paths to make the Makefile cleaner
PYTHON=./.venv/bin/python3
DATA_DIR=data
AB_TRAIN=$(DATA_DIR)/Assingment\ 1.2\ Files/part_ab_train.csv
AB_TEST=$(DATA_DIR)/Assingment\ 1.2\ Files/part_ab_test_public.csv
C_DATA_DIR=$(DATA_DIR)/part_c

# Default target
all: help

# -----------------------------------------------------------------------------
# Part A: Logistic Regression using SGD / Mini-batch / Full-batch
# -----------------------------------------------------------------------------
run_part_a_sgd:
	@echo "Running Part A (SGD)..."
	$(PYTHON) part_a.py "$(AB_TRAIN)" "$(AB_TEST)" sgd predictions_sgd.txt weights_sgd.txt

run_part_a_mini:
	@echo "Running Part A (Mini-batch)..."
	$(PYTHON) part_a.py "$(AB_TRAIN)" "$(AB_TEST)" mini_batch predictions_mini_batch.txt weights_mini_batch.txt

run_part_a_full:
	@echo "Running Part A (Full-batch)..."
	$(PYTHON) part_a.py "$(AB_TRAIN)" "$(AB_TEST)" full_batch predictions_full_batch.txt weights_full_batch.txt

run_part_a: run_part_a_sgd run_part_a_mini run_part_a_full

# -----------------------------------------------------------------------------
# Part B: Handling Class Imbalance
# -----------------------------------------------------------------------------
run_part_b_baseline:
	@echo "Running Part B (Baseline)..."
	$(PYTHON) part_b.py "$(AB_TRAIN)" "$(AB_TEST)" baseline predictions_baseline.txt weights_baseline.txt

run_part_b_classweight:
	@echo "Running Part B (Class Weighting)..."
	$(PYTHON) part_b.py "$(AB_TRAIN)" "$(AB_TEST)" class_weight predictions_classweight.txt weights_classweight.txt

run_part_b_classweight2:
	@echo "Running Part B (Class Weighting 2)..."
	$(PYTHON) part_b.py "$(AB_TRAIN)" "$(AB_TEST)" class_weight2 predictions_classweight2.txt weights_classweight2.txt

run_part_b_focal:
	@echo "Running Part B (Focal Loss)..."
	$(PYTHON) part_b.py "$(AB_TRAIN)" "$(AB_TEST)" focal predictions_focal.txt weights_focal.txt

run_part_b: run_part_b_baseline run_part_b_classweight run_part_b_classweight2 run_part_b_focal

# -----------------------------------------------------------------------------
# Part C: Patient-Level Atrial Fibrillation Detection
# -----------------------------------------------------------------------------
run_part_c:
	@echo "Running Part C Pipeline (Training & Prediction)..."
	$(PYTHON) part_c.py "$(C_DATA_DIR)" model.pkl final_features.csv

eval_part_c:
	@echo "================================================================"
	@echo "Running local evaluation using the TA's script on Validation Data"
	@echo "================================================================"
	@rm -rf "$(C_DATA_DIR)/temp_eval"
	@mkdir -p "$(C_DATA_DIR)/temp_eval"
	@cp "$(C_DATA_DIR)/train.csv" "$(C_DATA_DIR)/temp_eval/train.csv"
	@cp "$(C_DATA_DIR)/val.csv" "$(C_DATA_DIR)/temp_eval/val.csv"
	@cp "$(C_DATA_DIR)/val.csv" "$(C_DATA_DIR)/temp_eval/test.csv"
	$(PYTHON) part_c.py "$(C_DATA_DIR)/temp_eval" model.pkl val_features_for_eval.csv
	$(PYTHON) "$(DATA_DIR)/Assingment 1.2 Files/evaluate_partc.py" "$(C_DATA_DIR)/temp_eval/test.csv" model.pkl val_features_for_eval.csv
	@rm -rf "$(C_DATA_DIR)/temp_eval" val_features_for_eval.csv

kaggle_part_c:
	@echo "================================================================"
	@echo "Generating Kaggle Submission (train+val mode, toggle in part_c.py)"
	@echo "================================================================"
	$(PYTHON) part_c.py "$(C_DATA_DIR)" model.pkl final_features.csv
	$(PYTHON) "$(DATA_DIR)/Assingment 1.2 Files/partc_kaggle.py" model.pkl final_features.csv kaggle_submission.csv
	@echo "✓ kaggle_submission.csv ready!"


# -----------------------------------------------------------------------------
# Utility Targets
# -----------------------------------------------------------------------------
clean:
	@echo "Cleaning up generated predictions, weights, and model files..."
	rm -f predictions_*.txt weights_*.txt model.pkl final_features.csv submission.csv kaggle_submission.csv *.png
	rm -rf __pycache__ .pytest_cache

help:
	@echo "Available commands:"
	@echo "  make clean          - Remove all generated output files"
	@echo "  make run_part_a     - Run Part A for all solvers (SGD, Mini, Full)"
	@echo "  make run_part_b     - Run Part B for all imbalance handlers"
	@echo "  make run_part_c     - Train the Part C model and generate test outputs"
	@echo "  make eval_part_c    - Train and evaluate the Part C model"

