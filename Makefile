PYTHON := .venv/bin/python

TRAIN := data/e4_hr_train_downsampled.csv
TEST  := data/e4_hr_test_downsampled.csv

FOLDS := data/train_5fold.txt
REGULARIZATION := data/regularization.txt

SRC := src
EVAL := evaluation


# ============================================================
# Part (a) — Official evaluation
# ============================================================

.PHONY: eval-a
eval-a:
	$(PYTHON) $(EVAL)/eval_a.py \
		$(SRC)/part_a.py \
		$(TRAIN) \
		$(TEST)

.PHONY: check-a
check-a: eval-a


# ============================================================
# Part (b) — Official evaluation
# ============================================================

.PHONY: eval-b
eval-b:
	$(PYTHON) $(EVAL)/eval_b.py \
		$(SRC)/part_b.py \
		$(TRAIN) \
		$(TEST) \
		$(FOLDS) \
		$(REGULARIZATION)

.PHONY: check-b
check-b: eval-b


# ============================================================
# Part (c) — Official evaluation
# ============================================================

.PHONY: eval-c
eval-c:
	$(PYTHON) $(EVAL)/eval_c.py \
		$(SRC)/part_c.py \
		$(TRAIN) \
		$(TEST)

.PHONY: check-c
check-c: eval-c


# ============================================================
# Part (d)
# ============================================================

# Set these after confirming the exact Part (d) evaluator interface.
D_MODEL := model_d.pkl
D_FEATURES := features_d.npy
D_LABELS := data/public_test

.PHONY: eval-d
eval-d:
	$(PYTHON) $(EVAL)/eval_d.py \
		--model $(D_MODEL) \
		--features $(D_FEATURES) \
		--labels $(D_LABELS)

.PHONY: check-d
check-d: eval-d


# ============================================================
# Evaluate all
# ============================================================

.PHONY: evaluate
evaluate: eval-a eval-b eval-c
	@echo ""
	@echo "=========================================="
	@echo "All CSV-based evaluations complete."
	@echo "=========================================="


# ============================================================
# Help
# ============================================================

.PHONY: help
help:
	@echo "Available commands:"
	@echo ""
	@echo "  make eval-a    Official Part (a) evaluation"
	@echo "  make eval-b    Official Part (b) evaluation"
	@echo "  make eval-c    Official Part (c) evaluation"
	@echo "  make eval-d    Official Part (d) evaluation"
	@echo ""
	@echo "  make check-a   Same as eval-a"
	@echo "  make check-b   Same as eval-b"
	@echo "  make check-c   Same as eval-c"
	@echo "  make check-d   Same as eval-d"
	@echo ""
	@echo "  make evaluate  Evaluate Parts A, B and C"


# ============================================================
# Clean generated files
# ============================================================

.PHONY: clean
clean:
	@echo "Cleaning generated files..."

	# Prediction files
	find . -type f \( \
		-name "predictions*.txt" \
		-o -name "predictions*.csv" \
	\) -delete

	# Generated model / feature artifacts
	find . -type f \( \
		-name "*.pkl" \
		-o -name "*.npy" \
		-o -name "*.joblib" \
	\) -delete

	# Assignment-generated text files
	find . -type f \( \
		-name "weights.txt" \
		-o -name "bestlambda.txt" \
		-o -name "crossvalidation_errors.txt" \
	\) -delete

	# Python cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

	# Pytest cache, if created
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

	# Misc generated Python bytecode
	find . -type f -name "*.pyc" -delete

	@echo "Clean complete."