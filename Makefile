PYTHON := .venv/bin/python

TRAIN := data/e4_hr_train_downsampled.csv
TEST := data/e4_hr_test_downsampled.csv
FOLDS := data/train_5fold.txt
REG := data/regularization.txt

SRC := src

.PHONY: help test run-a run-b run-c run-all \
        evaluate-a evaluate-b evaluate-c evaluate \
        clean clean-predictions clean-cache

help:
	@echo "COL774 Assignment 1"
	@echo
	@echo "Usage:"
	@echo "  make test          - syntax-check Python files"
	@echo "  make run-a         - run Part (a)"
	@echo "  make run-b         - run Part (b)"
	@echo "  make run-c         - run Part (c)"
	@echo "  make run-all       - run Parts (a), (b), (c)"
	@echo "  make evaluate-a    - evaluate Part (a)"
	@echo "  make evaluate-b    - evaluate Part (b)"
	@echo "  make evaluate-c    - evaluate Part (c)"
	@echo "  make evaluate      - evaluate all existing predictions"
	@echo "  make clean         - remove generated outputs"
	@echo "  make clean-cache   - remove Python cache files"


test:
	$(PYTHON) -m py_compile \
		$(SRC)/part_a.py \
		$(SRC)/part_b.py \
		$(SRC)/part_c.py \
		$(SRC)/run_experiments.py

run-a:
	$(PYTHON) $(SRC)/run_experiments.py a

run-b:
	$(PYTHON) $(SRC)/run_experiments.py b

run-c:
	$(PYTHON) $(SRC)/run_experiments.py c

run-all:
	$(PYTHON) $(SRC)/run_experiments.py all


evaluate-a:
	$(PYTHON) $(SRC)/evaluate.py \
		$(TEST) \
		predictions_a.txt \
		"Part A - OLS"


evaluate-b:
	$(PYTHON) $(SRC)/evaluate.py \
		$(TEST) \
		predictions_b.txt \
		"Part B - Ridge"


evaluate-c:
	$(PYTHON) $(SRC)/evaluate.py \
		$(TEST) \
		predictions_c.txt \
		"Part C - Feature Engineering"


evaluate:
	$(PYTHON) $(SRC)/evaluate.py \
		$(TEST) \
		predictions_a.txt \
		predictions_b.txt \
		predictions_c.txt


clean-predictions:
	rm -f \
		predictions_a.txt \
		predictions_b.txt \
		predictions_c.txt


clean:
	rm -f \
		predictions_a.txt \
		predictions_b.txt \
		predictions_c.txt \
		weights_a.txt \
		weights_b.txt \
		bestlambda.txt \
		crossvalidation_errors.txt


clean-cache:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

